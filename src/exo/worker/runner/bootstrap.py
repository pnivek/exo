import atexit
import os
from pathlib import Path

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger

# tiktoken vocab files that openai_harmony's Rust backend needs.
# The Rust TLS stack (rustls/webpki-roots) can't verify Azure certs on
# some platforms, so we pre-download with Python's urllib (which uses the
# system CA store) and point TIKTOKEN_ENCODINGS_BASE at the local cache.
_TIKTOKEN_VOCAB_URL = "https://openaipublic.blob.core.windows.net/encodings"
_TIKTOKEN_VOCAB_FILES = ["o200k_base.tiktoken"]


def _ensure_tiktoken_vocab_cached() -> None:
    """Pre-download tiktoken vocab files so the Rust backend finds them locally.

    Sets TIKTOKEN_ENCODINGS_BASE to the local cache directory.  If the files
    are already cached this is a no-op.
    """
    if os.environ.get("TIKTOKEN_ENCODINGS_BASE"):
        return  # user already set a custom path

    cache_dir = Path.home() / ".cache" / "exo" / "tiktoken-encodings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in _TIKTOKEN_VOCAB_FILES:
        target = cache_dir / filename
        if target.exists():
            continue
        url = f"{_TIKTOKEN_VOCAB_URL}/{filename}"
        logger.info(f"Downloading tiktoken vocab: {url}")
        try:
            import urllib.request

            urllib.request.urlretrieve(url, str(target))  # noqa: S310
            logger.info(
                f"Cached tiktoken vocab: {target} ({target.stat().st_size} bytes)"
            )
        except Exception as exc:
            logger.warning(f"Failed to download tiktoken vocab {url}: {exc}")
            return  # don't set the env var if download failed

    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(cache_dir)
    logger.info(f"TIKTOKEN_ENCODINGS_BASE={cache_dir}")


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    def _cleanup_gpu() -> None:
        """Release GPU memory on runner exit (runs even on unhandled exceptions)."""
        try:
            import mlx.core as mx  # noqa: PLC0415

            mx.clear_cache()
        except Exception:
            pass

    atexit.register(_cleanup_gpu)

    _ensure_tiktoken_vocab_cached()

    # Increase CUDA graph cache for long-context TP prefill workloads.
    # Default (400) causes cache thrashing at 16K+ tokens with tensor parallelism.
    if "MLX_CUDA_GRAPH_CACHE_SIZE" not in os.environ:
        os.environ["MLX_CUDA_GRAPH_CACHE_SIZE"] = "2000"

    import mlx.core as mx

    if mx.metal.is_available():
        fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
        if fast_synch_override != "off":
            os.environ["MLX_METAL_FAST_SYNCH"] = "1"
        else:
            os.environ["MLX_METAL_FAST_SYNCH"] = "0"
        logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")
    else:
        logger.info("Metal not available, skipping MLX_METAL_FAST_SYNCH")
        # On CUDA unified-memory devices (GB10), the default 95% cache limit
        # causes freed buffers to stay resident in the OS page cache
        # indefinitely.  nvidia-smi can't track memory on these devices, so
        # the OS sees it as "used" and memory accumulates across requests.
        # Disable buffer caching so every free returns memory to the system.
        old_limit = mx.set_cache_limit(0)
        logger.info(f"CUDA buffer cache disabled (was {old_limit / 1024**3:.1f} GB)")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import main
        else:
            from exo.worker.runner.llm_inference.runner import main

        main(bound_instance, event_sender, task_receiver, cancel_receiver)

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
