import ctypes
import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import mlx.core as mx
from anyio import WouldBlock
from mlx_lm.generate import GenerationResponse as MlxGenerationResponse
from mlx_lm.generate import stream_generate
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    PromptTokensDetails,
    Usage,
)
from exo.shared.models.model_cards import ModelTask
from exo.shared.types.chunks import (
    ErrorChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.mlx import Model
from exo.shared.types.tasks import (
    ConnectToGroup,
    DisaggDecode,
    DisaggPrefill,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TensorParallelDisaggPrefill,
    TextGeneration,
)
from exo.shared.types.worker.instances import (
    BoundInstance,
    DisaggregatedInstance,
    TensorPrefillDisaggInstance,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,
    load_mlx_items,
    mx_barrier,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    InferenceGenerator,
    SequentialGenerator,
)

from .batch_generator import Cancelled, Finished
from .tool_parsers import make_mlx_parser


def _align_received_caches(
    model: Model,
    received_caches: list[Any],
) -> list[Any]:
    """Align received KV caches to the model's expected cache types.

    Models like gpt-oss use interleaved attention — some layers expect
    RotatingKVCache (sliding window) while others use plain KVCache.
    The KV transfer receiver creates all caches as plain KVCache, which
    makes sliding-attention layers attend to the entire sequence instead
    of just the window — causing the GPU to hang on long contexts.
    """
    if not hasattr(model, "make_cache"):
        return received_caches

    from mlx_lm.models.cache import KVCache, RotatingKVCache

    expected_caches: list[Any] = model.make_cache()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if len(expected_caches) != len(received_caches):  # pyright: ignore[reportUnknownArgumentType]
        logger.warning(
            f"Cache count mismatch: model expects {len(expected_caches)}, "  # pyright: ignore[reportUnknownArgumentType]
            f"received {len(received_caches)}. Skipping alignment."
        )
        return received_caches

    aligned: list[Any] = []
    converted_count = 0
    for expected, received_any in zip(  # pyright: ignore[reportAny]
        expected_caches,  # pyright: ignore[reportUnknownArgumentType]
        received_caches,
        strict=True,
    ):
        if not isinstance(expected, RotatingKVCache):
            aligned.append(received_any)
            continue

        received = cast(KVCache, received_any)

        # Extract K/V from the received plain KVCache.
        keys, values = received.state
        assert keys is not None and values is not None
        seq_len: int = keys.shape[2]
        max_size = expected.max_size

        if seq_len <= max_size:
            # Sequence fits in the window — just wrap in RotatingKVCache.
            rotating = RotatingKVCache(max_size=max_size, keep=expected.keep)
            rotating.state = (keys, values)
            rotating.offset = received.offset
            rotating._idx = seq_len
        else:
            # Trim to the last max_size tokens for the sliding window.
            trimmed_keys = keys[:, :, -max_size:, :]
            trimmed_values = values[:, :, -max_size:, :]
            rotating = RotatingKVCache(max_size=max_size, keep=expected.keep)
            rotating.state = (trimmed_keys, trimmed_values)
            rotating.offset = received.offset
            # Buffer is full — next write wraps to keep position.
            rotating._idx = max_size

        aligned.append(rotating)
        converted_count += 1

    if converted_count > 0:
        logger.info(
            f"Aligned {converted_count}/{len(received_caches)} caches "
            f"to RotatingKVCache (sliding window)"
        )

    return aligned


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Shutdown = "Shutdown"


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.device_rank = self.shard_metadata.device_rank
        self.kv_transfer_server = None  # persistent KV receiver for disagg decode

        logger.info("hello from the runner")
        self._configure_cuda_memory_pool()
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()

        self.generator: Builder | InferenceGenerator = Builder(
            self.model_id, self.event_sender, self.cancel_receiver
        )

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            TextGeneration,
        ] = {}

        logger.info("runner created")
        self.update_status(RunnerIdle())

    def _configure_cuda_memory_pool(self) -> None:
        """Cache a handle to the default CUDA memory pool for later trimming."""
        if mx.default_device() != mx.Device(mx.gpu):
            self._cuda_pool = None
            return
        try:
            self._libcudart = ctypes.CDLL("libcudart.so")
            pool = ctypes.c_void_p()
            get_pool = self._libcudart.cudaDeviceGetDefaultMemPool
            get_pool.restype = ctypes.c_int
            if get_pool(ctypes.byref(pool), 0) != 0:
                logger.debug("cudaDeviceGetDefaultMemPool failed")
                self._cuda_pool = None
                return
            self._cuda_pool = pool
            logger.info("CUDA memory pool handle acquired for post-prefill trimming")
        except Exception as exc:
            logger.debug(f"Could not get CUDA memory pool: {exc}")
            self._cuda_pool = None

    def _log_cuda_mem(self, label: str) -> None:
        """Log CUDA free/total memory via cudaMemGetInfo."""
        if self._cuda_pool is None:
            return
        try:
            free = ctypes.c_size_t()
            total = ctypes.c_size_t()
            rc = cast(
                int,
                self._libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
            )
            if rc == 0:
                f_gb = free.value / (1024**3)
                t_gb = total.value / (1024**3)
                u_gb = t_gb - f_gb
                logger.info(
                    f"[CUDA MEM {label}] used={u_gb:.2f} GB  free={f_gb:.2f} GB  total={t_gb:.2f} GB"
                )
        except Exception:
            pass

    def _reclaim_gpu_memory(self) -> None:
        """Reclaim GPU memory after prefill by flushing all caches.

        The correct order is critical:
        1. synchronize — flush GPU completion handlers that hold shared_ptr
           temporaries to buffers.  Until these run, buffers are "active"
           and invisible to clear_cache.
        2. clear_cache — move freed buffers from MLX's buffer cache to the
           CUDA allocator (cudaFree / cudaFreeAsync).
        3. clear_graph_caches — destroy cached CUDA graph executables whose
           workspace memory is pinned by the CUDA runtime.  This also calls
           cudaMemPoolTrimTo internally.
        """
        # Step 1: ensure all GPU work is done and completion handlers have
        # released their temporary references to arrays.
        mx.synchronize()

        # Step 2: free MLX's buffer cache (now populated with released buffers).
        cache_before = mx.get_cache_memory()
        active_before = mx.get_active_memory()
        mx.clear_cache()
        cache_after = mx.get_cache_memory()
        active_after = mx.get_active_memory()
        logger.info(
            f"[MLX MEM] active: {active_before / (1024**3):.2f} -> {active_after / (1024**3):.2f} GB, "
            f"cache: {cache_before / (1024**3):.2f} -> {cache_after / (1024**3):.2f} GB"
        )

        # Step 3: destroy cached CUDA graph execs to free workspace memory.
        # mx.cuda.clear_graph_caches() also trims memory pools internally.
        try:
            mx.cuda.clear_graph_caches()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            logger.info("CUDA graph caches cleared")
        except Exception as e:
            # Binding not available in this MLX build — fall back to pool trim.
            logger.warning(f"clear_graph_caches unavailable: {e}")
            if self._cuda_pool is not None:
                self._libcudart.cudaMemPoolTrimTo(self._cuda_pool, ctypes.c_size_t(0))

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        with self.task_receiver:
            for task in self.task_receiver:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.handle_first_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                assert isinstance(self.generator, Builder)
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                self.generator.group = initialize_mlx(self.bound_instance)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if isinstance(self.generator, Builder) and (
                (
                    isinstance(self.current_status, RunnerConnected)
                    and self.generator.group is not None
                )
                or (
                    isinstance(self.current_status, RunnerIdle)
                    and self.generator.group is None
                )
            ):
                total_layers = (
                    self.shard_metadata.end_layer - self.shard_metadata.start_layer
                )
                logger.info("runner loading")

                self.update_status(
                    RunnerLoading(layers_loaded=0, total_layers=total_layers)
                )
                self.acknowledge_task(task)

                def on_model_load_timeout() -> None:
                    self.update_status(
                        RunnerFailed(error_message="Model loading timed out")
                    )
                    time.sleep(0.5)

                def on_layer_loaded(layers_loaded: int, total: int) -> None:
                    self.update_status(
                        RunnerLoading(layers_loaded=layers_loaded, total_layers=total)
                    )

                assert (
                    ModelTask.TextGeneration in self.shard_metadata.model_card.tasks
                ), f"Incorrect model task(s): {self.shard_metadata.model_card.tasks}"
                self.generator.inference_model, self.generator.tokenizer = (
                    load_mlx_items(
                        self.bound_instance,
                        self.generator.group,
                        on_timeout=on_model_load_timeout,
                        on_layer_loaded=on_layer_loaded,
                    )
                )

                self.generator = self.generator.build()

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                logger.info("runner loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                assert isinstance(self.generator, InferenceGenerator)
                logger.info("runner warming up")

                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                self.generator.warmup()

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                # Start persistent KV transfer server for disagg decode runners.
                # The server binds once and stays listening across requests,
                # so the prefill node can always connect immediately.
                instance = self.bound_instance.instance
                if isinstance(
                    instance, (DisaggregatedInstance, TensorPrefillDisaggInstance)
                ):
                    from exo.worker.engines.mlx.kv_transfer import (
                        KV_TRANSFER_PORT,
                        KVTransferServer,
                    )

                    is_decode_runner = isinstance(instance, DisaggregatedInstance) and (
                        self.bound_instance.bound_node_id == instance.decode_node_id
                    )
                    is_tp_decode_runner = isinstance(
                        instance, TensorPrefillDisaggInstance
                    ) and (self.bound_instance.bound_node_id == instance.decode_node_id)
                    if is_decode_runner or is_tp_decode_runner:
                        self.kv_transfer_server = KVTransferServer(KV_TRANSFER_PORT)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())
                logger.info("runner ready")

            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case DisaggPrefill() if isinstance(self.current_status, RunnerReady):
                self._handle_disagg_prefill(task)

            case TensorParallelDisaggPrefill() if isinstance(
                self.current_status, RunnerReady
            ):
                self._handle_tp_disagg_prefill(task)

            case DisaggDecode() if isinstance(self.current_status, RunnerReady):
                self._handle_disagg_decode(task)

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def shutdown(self, task: Task):
        logger.info("runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        if self.kv_transfer_server is not None:
            self.kv_transfer_server.close()
            self.kv_transfer_server = None
        if isinstance(self.generator, InferenceGenerator):
            self.generator.close()
        mx.clear_cache()
        import gc

        gc.collect()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def submit_text_generation(self, task: TextGeneration):
        assert isinstance(self.generator, InferenceGenerator)
        self.active_tasks[task.task_id] = task
        self.generator.submit(task)

    def handle_generation_tasks(self, starting_task: TextGeneration):
        assert isinstance(self.current_status, RunnerReady)
        assert isinstance(self.generator, InferenceGenerator)

        logger.info(f"received chat request: {starting_task}")
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        self.submit_text_generation(starting_task)

        while self.active_tasks:
            results = self.generator.step()

            finished: list[TaskId] = []
            for task_id, result in results:
                match result:
                    case Cancelled():
                        finished.append(task_id)
                    case Finished():
                        self.send_task_status(task_id, TaskStatus.Complete)
                        finished.append(task_id)
                    case _:
                        self.send_response(
                            result, self.active_tasks[task_id].command_id
                        )

            for task_id in finished:
                self.active_tasks.pop(task_id, None)

            try:
                task = self.task_receiver.receive_nowait()

                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)

                match task:
                    case TextGeneration():
                        self.acknowledge_task(task)
                        self.submit_text_generation(task)
                    case Shutdown():
                        self.shutdown(task)
                        return ExitCode.Shutdown
                    case _:
                        raise ValueError(
                            f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                        )

            except WouldBlock:
                pass

        self.update_status(RunnerReady())
        logger.info("runner ready")

        return ExitCode.AllTasksComplete

    def send_response(
        self,
        response: GenerationResponse | ToolCallResponse,
        command_id: CommandId,
    ):
        match response:
            case GenerationResponse():
                if self.device_rank == 0 and response.finish_reason == "error":
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ErrorChunk(
                                error_message=response.text,
                                model=self.model_id,
                            ),
                        )
                    )

                elif self.device_rank == 0:
                    assert response.finish_reason not in (
                        "error",
                        "tool_calls",
                        "function_call",
                    )
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=TokenChunk(
                                model=self.model_id,
                                text=response.text,
                                token_id=response.token,
                                usage=response.usage,
                                finish_reason=response.finish_reason,
                                stats=response.stats,
                                logprob=response.logprob,
                                top_logprobs=response.top_logprobs,
                                is_thinking=response.is_thinking,
                            ),
                        )
                    )
            case ToolCallResponse():
                if self.device_rank == 0:
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ToolCallChunk(
                                tool_calls=response.tool_calls,
                                model=self.model_id,
                                usage=response.usage,
                                stats=response.stats,
                            ),
                        )
                    )

    def _handle_disagg_prefill(self, task: DisaggPrefill) -> None:
        """Handle disaggregated prefill: prefill on this node, send KV cache to decode node."""
        assert isinstance(self.generator, (SequentialGenerator, BatchGenerator))
        logger.info(f"received disaggregated prefill request: {task}")
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)

        inference_model = self.generator.model
        tokenizer = self.generator.tokenizer

        try:
            from mlx_lm.sample_utils import make_sampler

            from exo.worker.engines.mlx.cache import (
                encode_prompt,
                make_kv_cache,
            )
            from exo.worker.engines.mlx.constants import (
                DISAGG_REPREFILL_TOKENS,
                HARMONY_CHANNEL_TOKEN_ID,
            )
            from exo.worker.engines.mlx.kv_transfer import (
                send_kv_cache_per_layer_sync,
                send_kv_cache_pipelined_sync,
            )
            from exo.worker.engines.mlx.utils_mlx import apply_chat_template

            prompt = apply_chat_template(tokenizer, task.task_params)
            all_prompt_tokens = encode_prompt(tokenizer, prompt)

            # Template-aware: append the <|channel|> token so the
            # decode node's first generated token is the channel
            # name, not a protocol marker that MoE routing
            # divergence might corrupt.
            if isinstance(inference_model, GptOssModel):
                all_prompt_tokens = mx.concatenate(
                    [
                        all_prompt_tokens,
                        mx.array([HARMONY_CHANNEL_TOKEN_ID]),
                    ]
                )

            caches = make_kv_cache(model=inference_model)
            sampler = make_sampler(
                temp=task.task_params.temperature
                if task.task_params.temperature is not None
                else 0.7,
            )

            reprefill = min(DISAGG_REPREFILL_TOKENS, len(all_prompt_tokens))
            reprefill = max(reprefill, 2)
            last_tokens = all_prompt_tokens[-reprefill:]
            t_pipelined_start = time.monotonic()

            # Use per-layer streaming for gpt_oss models (streams each layer's
            # KV as soon as it completes, overlapping network with compute).
            # Fall back to per-chunk pipelining for other model types.
            if isinstance(inference_model, GptOssModel):
                prefill_tps, num_tokens = send_kv_cache_per_layer_sync(
                    host=task.decode_node_host,
                    port=task.decode_node_port,
                    model=inference_model,
                    tokenizer=tokenizer,
                    prompt_tokens=all_prompt_tokens[:-1],
                    last_tokens=last_tokens,
                    cache=caches,
                    sampler=sampler,
                )
            else:
                prefill_tps, num_tokens = send_kv_cache_pipelined_sync(
                    host=task.decode_node_host,
                    port=task.decode_node_port,
                    model=inference_model,
                    tokenizer=tokenizer,
                    prompt_tokens=all_prompt_tokens[:-1],
                    last_tokens=last_tokens,
                    cache=caches,
                    sampler=sampler,
                )
            t_pipelined_end = time.monotonic()
            logger.info(
                f"DISAGG_TIMING pipelined_total_ms={(t_pipelined_end - t_pipelined_start) * 1000:.1f} "
                f"prefill_tps={prefill_tps:.1f} num_tokens={num_tokens}"
            )

            # Free KV cache and intermediate tensors to prevent
            # GPU memory accumulation across consecutive requests.
            self._log_cuda_mem("before-cleanup")
            del caches
            import gc

            gc.collect()
            self._reclaim_gpu_memory()
            self._log_cuda_mem("after-cleanup")

        except Exception as e:
            logger.opt(exception=e).error("DisaggPrefill failed, recovering runner")
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=ErrorChunk(
                            model=self.model_id,
                            finish_reason="error",
                            error_message=str(e),
                        ),
                    )
                )
            import gc

            gc.collect()
            self._reclaim_gpu_memory()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerReady())
        logger.info("runner ready")

    def _handle_tp_disagg_prefill(self, task: TensorParallelDisaggPrefill) -> None:
        """Handle tensor-parallel disaggregated prefill."""
        assert isinstance(self.generator, (SequentialGenerator, BatchGenerator))
        logger.info(f"received tensor-parallel disagg prefill request: {task}")
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)

        inference_model = self.generator.model
        tokenizer = self.generator.tokenizer
        group = self.generator.group
        assert group is not None, "TP prefill requires distributed group"

        try:
            from mlx_lm.models.cache import KVCache as PlainKVCache
            from mlx_lm.sample_utils import make_sampler

            from exo.worker.engines.mlx.cache import (
                encode_prompt,
            )
            from exo.worker.engines.mlx.constants import (
                DISAGG_REPREFILL_TOKENS,
                HARMONY_CHANNEL_TOKEN_ID,
            )
            from exo.worker.engines.mlx.utils_mlx import apply_chat_template

            prompt = apply_chat_template(tokenizer, task.task_params)
            all_prompt_tokens = encode_prompt(tokenizer, prompt)

            if isinstance(inference_model, GptOssModel):
                all_prompt_tokens = mx.concatenate(
                    [
                        all_prompt_tokens,
                        mx.array([HARMONY_CHANNEL_TOKEN_ID]),
                    ]
                )

            reprefill = min(DISAGG_REPREFILL_TOKENS, len(all_prompt_tokens))
            reprefill = max(reprefill, 2)
            last_tokens = all_prompt_tokens[-reprefill:]

            caches = [PlainKVCache() for _ in inference_model.layers]
            sampler = make_sampler(
                temp=task.task_params.temperature
                if task.task_params.temperature is not None
                else 0.7,
            )

            # Sync NCCL state before prefill — upstream's warmup_inference
            # added an all_gather that advances the NCCL op counter. Without
            # this barrier, ranks can desync across consecutive requests.
            mx_barrier(group)

            instance = self.bound_instance.instance
            is_kv_sender = (
                isinstance(instance, TensorPrefillDisaggInstance)
                and self.bound_instance.bound_node_id == instance.kv_sender_node_id
            )

            t_pipelined_start = time.monotonic()

            # Per-layer TP prefill with NCCL all_gather — eliminates
            # TCP relay bottleneck (~250-600ms) that caused GPU idle gaps.
            from exo.worker.engines.mlx.kv_transfer import (
                send_kv_cache_per_layer_tp_sync,
            )

            prefill_tps, num_tokens = send_kv_cache_per_layer_tp_sync(
                host=task.decode_node_host,
                port=task.decode_node_port,
                model=cast(Model, inference_model),
                tokenizer=tokenizer,
                prompt_tokens=all_prompt_tokens[:-1],
                last_tokens=last_tokens,
                cache=caches,
                sampler=sampler,
                group=group,
                is_kv_sender=is_kv_sender,
            )

            t_pipelined_end = time.monotonic()
            logger.info(
                f"DISAGG_TIMING tp_prefill_total_ms={(t_pipelined_end - t_pipelined_start) * 1000:.1f} "
                f"prefill_tps={prefill_tps:.1f} num_tokens={num_tokens}"
            )

            # Free KV cache and intermediate tensors to prevent
            # GPU memory accumulation across consecutive requests.
            self._log_cuda_mem("before-cleanup")
            del caches
            import gc

            gc.collect()
            self._reclaim_gpu_memory()
            self._log_cuda_mem("after-cleanup")

        except Exception as e:
            logger.opt(exception=e).error(
                "TensorParallelDisaggPrefill failed, recovering runner"
            )
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=ErrorChunk(
                            model=self.model_id,
                            finish_reason="error",
                            error_message=str(e),
                        ),
                    )
                )
            import gc

            gc.collect()
            self._reclaim_gpu_memory()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerReady())
        logger.info("runner ready")

    def _handle_disagg_decode(self, task: DisaggDecode) -> None:
        """Handle disaggregated decode: receive KV cache and generate tokens."""
        assert isinstance(self.generator, (SequentialGenerator, BatchGenerator))
        logger.info(
            f"received disaggregated decode request, waiting for KV cache on port {task.kv_transfer_port}"
        )
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)

        inference_model = self.generator.model
        tokenizer = self.generator.tokenizer

        try:
            from mlx_lm.sample_utils import make_sampler

            from exo.worker.engines.mlx.constants import (
                DISAGG_REPREFILL_TOKENS,
                HARMONY_CHANNEL_TOKEN_ID,
                KV_BITS,
                KV_GROUP_SIZE,
                MAX_TOKENS,
            )
            from exo.worker.engines.mlx.kv_transfer import (
                receive_kv_cache_auto_sync,
            )
            from exo.worker.engines.mlx.utils_mlx import apply_chat_template
            from exo.worker.runner.llm_inference.model_output_parsers import (
                parse_gpt_oss,
                parse_thinking_models,
            )

            t_kv_wait_start = time.monotonic()
            if self.kv_transfer_server is not None:
                received_caches, last_tokens = self.kv_transfer_server.receive()
            else:
                received_caches, last_tokens = receive_kv_cache_auto_sync(
                    task.kv_transfer_port
                )
            t_kv_wait_end = time.monotonic()
            logger.info(
                f"DISAGG_TIMING decode_kv_wait_ms={(t_kv_wait_end - t_kv_wait_start) * 1000:.1f} "
                f"layers={len(received_caches)}"
            )

            # Align received caches to the model's expected cache types.
            # Models like gpt-oss use interleaved attention: some layers
            # use RotatingKVCache (sliding window) while others use KVCache
            # (full attention).  The receiver creates all caches as plain
            # KVCache, which makes sliding-attention layers attend to the
            # entire sequence instead of just max_size tokens — causing
            # Metal to hang on long contexts.
            t_align_start = time.monotonic()
            received_caches = _align_received_caches(inference_model, received_caches)
            t_align_end = time.monotonic()
            logger.info(
                f"DISAGG_TIMING decode_align_caches_ms={(t_align_end - t_align_start) * 1000:.1f}"
            )

            sampler = make_sampler(
                temp=task.task_params.temperature
                if task.task_params.temperature is not None
                else 0.7,
                top_p=task.task_params.top_p
                if task.task_params.top_p is not None
                else 1.0,
            )
            max_tokens = task.task_params.max_output_tokens or MAX_TOKENS

            t_decode_start = time.monotonic()

            prompt = apply_chat_template(tokenizer, task.task_params)
            prompt_token_count = len(tokenizer.encode(prompt))

            raw_stream = stream_generate(
                model=inference_model,
                tokenizer=tokenizer,
                prompt=last_tokens,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=received_caches,
                prefill_step_size=DISAGG_REPREFILL_TOKENS,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
            )
            gen: Generator[GenerationResponse | None, None, None] = (
                _wrap_stream_generate(
                    raw_stream,
                    t_decode_start,
                    prompt_tokens=prompt_token_count,
                )
            )

            if isinstance(inference_model, GptOssModel):

                def _prepend_channel(
                    stream: Generator[GenerationResponse | None, None, None],
                ) -> Generator[GenerationResponse | None, None, None]:
                    yield GenerationResponse(
                        text="<|channel|>",
                        token=HARMONY_CHANNEL_TOKEN_ID,
                        finish_reason=None,
                        usage=None,
                    )
                    yield from stream

                gen = _prepend_channel(gen)

            if tokenizer.has_thinking:
                gen = parse_thinking_models(
                    gen,
                    tokenizer.think_start,
                    tokenizer.think_end,
                    starts_in_thinking=False,
                )
            parsed: Generator[
                GenerationResponse | ToolCallResponse | None, None, None
            ] = gen
            if isinstance(inference_model, GptOssModel):
                parsed = parse_gpt_oss(gen)

            t_first_token: float | None = None
            decode_token_count = 0
            for response in parsed:
                if response is None:
                    continue

                decode_token_count += 1
                if t_first_token is None:
                    t_first_token = time.monotonic()
                    logger.info(
                        f"DISAGG_TIMING decode_ttfr_ms={(t_first_token - t_decode_start) * 1000:.1f} "
                        f"(includes reprefill of {len(last_tokens)} tokens)"
                    )

                if self.device_rank == 0:
                    self.send_response(response, task.command_id)
                if (
                    isinstance(response, GenerationResponse)
                    and response.finish_reason is not None
                ):
                    break

            t_decode_end = time.monotonic()
            if t_first_token is not None and decode_token_count > 1:
                steady_state_ms = (t_decode_end - t_first_token) * 1000
                steady_state_tps = (
                    (decode_token_count - 1) / (steady_state_ms / 1000)
                    if steady_state_ms > 0
                    else 0.0
                )
                total_decode_ms = (t_decode_end - t_decode_start) * 1000
                overall_tps = (
                    decode_token_count / (total_decode_ms / 1000)
                    if total_decode_ms > 0
                    else 0.0
                )
                logger.info(
                    f"DISAGG_TIMING decode_steady_state_tps={steady_state_tps:.1f} "
                    f"overall_tps={overall_tps:.1f} "
                    f"tokens={decode_token_count} "
                    f"steady_state_ms={steady_state_ms:.1f} "
                    f"total_decode_ms={total_decode_ms:.1f}"
                )

            del received_caches
            import gc

            gc.collect()
            mx.clear_cache()

        except Exception as e:
            logger.opt(exception=e).error("DisaggDecode failed, recovering runner")
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=ErrorChunk(
                            model=self.model_id,
                            finish_reason="error",
                            error_message=str(e),
                        ),
                    )
                )
            import gc

            gc.collect()
            mx.clear_cache()
            mx.synchronize()

        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerReady())
        logger.info("runner ready")


def _wrap_stream_generate(
    raw_stream: Generator[MlxGenerationResponse, None, None],
    t_decode_start: float,
    prompt_tokens: int = 0,
) -> Generator[GenerationResponse, None, None]:
    """Wrap raw mlx_lm stream_generate output into GenerationResponse objects.

    Also logs first-token timing for disaggregated decode and computes
    usage on the final response.
    """
    t_first_token: float | None = None
    for completion_tokens, out in enumerate(raw_stream, start=1):
        if t_first_token is None:
            t_first_token = time.monotonic()
            logger.info(
                f"DISAGG_TIMING decode_first_token_ms={(t_first_token - t_decode_start) * 1000:.1f}"
            )
        finish_reason: FinishReason | None = cast(
            FinishReason | None, out.finish_reason
        )
        usage: Usage | None = None
        if finish_reason is not None:
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(),
                completion_tokens_details=CompletionTokensDetails(),
            )
        yield GenerationResponse(
            text=out.text,
            token=out.token,
            finish_reason=finish_reason,
            usage=usage,
        )


@dataclass
class Builder:
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None

    def build(
        self,
    ) -> InferenceGenerator:
        assert self.model_id
        assert self.inference_model
        assert self.tokenizer

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)

        device_rank = 0 if self.group is None else self.group.rank()
        if os.environ.get("EXO_NO_BATCH"):
            logger.info("using SequentialGenerator (batching disabled)")
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
            )
        logger.info("using BatchGenerator")
        return BatchGenerator(
            model=self.inference_model,
            tokenizer=self.tokenizer,
            group=self.group,
            tool_parser=tool_parser,
            kv_prefix_cache=kv_prefix_cache,
            model_id=self.model_id,
            device_rank=device_rank,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
        )
