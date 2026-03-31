import ctypes
import os
import resource
import sys
import urllib.request
from pathlib import Path

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, VllmInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger

_TIKTOKEN_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings"
_TIKTOKEN_FILES = ["o200k_base.tiktoken", "cl100k_base.tiktoken"]

_CUDA_HOST_LIBS = ["libcuda.so.1", "libnvidia-ml.so.1"]
# NOTE: libnvidia-ptxjitcompiler.so.1 is intentionally NOT loaded here.
# Loading it with RTLD_GLOBAL conflicts with Triton 3.6's own ptxjit
# compilation pipeline, causing "Triton Error [CUDA]: initialization error"
# in the runner subprocess on SM121a (GB10 Blackwell).
_CUDA_HOST_SEARCH_DIRS = [
    Path("/usr/lib/aarch64-linux-gnu"),
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/lib64"),
    Path("/usr/lib"),
    Path("/usr/local/cuda/lib64"),
    Path("/usr/local/cuda/compat"),
]


def _ensure_tiktoken_encodings() -> None:
    if os.environ.get("TIKTOKEN_ENCODINGS_BASE"):
        return
    from exo.shared.constants import EXO_CACHE_HOME

    enc_dir = EXO_CACHE_HOME / "encodings"
    enc_dir.mkdir(parents=True, exist_ok=True)
    for fname in _TIKTOKEN_FILES:
        dest = enc_dir / fname
        if dest.exists():
            continue
        url = f"{_TIKTOKEN_BASE_URL}/{fname}"
        logger.info(f"Downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception:
            logger.warning(f"Failed to download {fname}, harmony encoding may fail")
            return
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(enc_dir)
    logger.info(f"Set TIKTOKEN_ENCODINGS_BASE={enc_dir}")


def _ensure_cuda_libs() -> None:
    if sys.platform != "linux":
        return
    for search_dir in _CUDA_HOST_SEARCH_DIRS:
        driver = search_dir / "libcuda.so.1"
        if not driver.exists():
            continue
        for lib_name in _CUDA_HOST_LIBS:
            lib_path = search_dir / lib_name
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                    logger.info(f"Loaded CUDA host lib: {lib_path}")
                except OSError:
                    logger.warning(f"Failed to load {lib_path}")
                    raise
        return


def _patch_flashinfer_triton_kernels() -> None:
    """Replace FlashInfer's Triton JIT kernels with pure-PyTorch equivalents.

    On SM121a (GB10 Blackwell), Triton's CUDA initialization fails in exo's
    runner subprocess with 'Triton Error [CUDA]: initialization error'.  This
    only affects processes spawned from exo's main process — standalone Triton
    works fine.  The root cause is unknown but is likely related to inherited
    process state from the parent (Rust/libp2p networking, async event loop).

    FlashInfer's ``_copy_page_indices_kernel`` is the only Triton kernel in the
    critical path.  It's a simple data-copy kernel that can be replaced with a
    PyTorch equivalent without any performance impact (it operates on index
    tensors, not large data arrays).
    """
    try:
        import torch

        def _copy_page_indices_pytorch(
            page_indices: torch.Tensor,
            block_table: torch.Tensor,
            block_table_stride: int,
            cu_num_blocks: torch.Tensor,
            BLOCK_SIZE: int,  # noqa: N803 — matches Triton signature
        ) -> None:
            """PyTorch replacement for FlashInfer's _copy_page_indices_kernel.

            For each request i, copies block_table[i, :num_blocks[i]] into
            page_indices[cu_num_blocks[i]:cu_num_blocks[i+1]].
            """
            num_reqs = cu_num_blocks.shape[0] - 1
            starts = cu_num_blocks[:num_reqs]
            ends = cu_num_blocks[1 : num_reqs + 1]
            counts = ends - starts

            for i in range(num_reqs):
                n = counts[i].item()
                if n > 0:
                    src = block_table[i, :n]
                    dst_start = starts[i].item()
                    page_indices[dst_start : dst_start + n] = src

        # Wrap to match the Triton kernel's launch API:
        #   _copy_page_indices_kernel[(num_reqs,)](page_indices, block_table, stride, cu_num_blocks, BLOCK_SIZE=...)
        class _TritonKernelShim:
            """Mimics Triton's grid-launch syntax: kernel[(grid,)](args, BLOCK_SIZE=N)."""

            def __getitem__(self, grid: tuple[int, ...]) -> "_TritonKernelShim._Launcher":
                return self._Launcher(grid)

            class _Launcher:
                def __init__(self, grid: tuple[int, ...]) -> None:
                    self.grid = grid

                def __call__(
                    self,
                    page_indices: torch.Tensor,
                    block_table: torch.Tensor,
                    block_table_stride: int,
                    cu_num_blocks: torch.Tensor,
                    BLOCK_SIZE: int = 128,  # noqa: N803
                ) -> None:
                    _copy_page_indices_pytorch(
                        page_indices, block_table, block_table_stride, cu_num_blocks, BLOCK_SIZE
                    )

        from vllm.v1.attention.backends import flashinfer as fi_mod

        fi_mod._copy_page_indices_kernel = _TritonKernelShim()  # type: ignore[attr-defined]
        logger.info("Patched FlashInfer _copy_page_indices_kernel with PyTorch equivalent (Triton bypass)")
    except Exception as e:
        logger.warning(f"Failed to patch FlashInfer Triton kernels: {e}")


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override != "off":
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    from exo.worker.engines.mlx.yarn_rope_patch import patch_yarn_rope

    patch_yarn_rope()

    from exo.worker.engines.mlx.gdn_softplus_patch import patch_gdn_softplus

    patch_gdn_softplus()

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if isinstance(bound_instance.instance, VllmInstance):
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
            os.environ["FASTSAFETENSORS_NOGDS"] = "1"
            # os.environ["VLLM_BATCH_INVARIANT"] = "1"
            # NOTE: _ensure_cuda_libs() intentionally NOT called.
            # Pre-loading libcuda.so.1/libnvidia-ml.so.1 with RTLD_GLOBAL
            # causes Triton's cuInit(0) to fail in this subprocess with
            # "Triton Error [CUDA]: initialization error". vLLM and PyTorch
            # load CUDA libs correctly on their own.
            _patch_flashinfer_triton_kernels()
            _ensure_tiktoken_encodings()
            from exo.shared.constants import EXO_MODELS_DIR
            from exo.worker.runner.llm_inference.runner import Runner, VllmBuilder

            model_id = bound_instance.bound_shard.model_card.model_id
            builder = VllmBuilder(
                model_id=model_id,
                model_path=str(EXO_MODELS_DIR / model_id.normalize()),
                trust_remote_code=bound_instance.bound_shard.model_card.trust_remote_code,
                cancel_receiver=cancel_receiver,
                event_sender=event_sender,
            )
            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver, builder
            )
            runner.main()
        elif bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import Runner as ImageRunner

            runner = ImageRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        else:
            from exo.worker.runner.llm_inference.runner import MlxBuilder, Runner

            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender,
                cancel_receiver=cancel_receiver,
            )
            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver, builder
            )
            runner.main()

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
