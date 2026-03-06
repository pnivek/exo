import ctypes
import gc
import resource
import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import mlx.core as mx
from anyio import WouldBlock
from mlx_lm.generate import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper

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
from exo.shared.types.text_generation import TextGenerationTaskParams
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
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    InferenceGenerator,
    SequentialGenerator,
)

from .batch_generator import Cancelled, Finished
from .tool_parsers import ToolParser, make_mlx_parser


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

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.device_rank = self.shard_metadata.device_rank

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
        self.kv_transfer_server: Any = None

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
                logger.info(
                    f"model has_tool_calling={self.generator.tokenizer.has_tool_calling} using tokens {self.generator.tokenizer.tool_call_start}, {self.generator.tokenizer.tool_call_end}"
                )
                tok = self.generator.tokenizer
                if tok.tool_call_start and tok.tool_call_end and tok.tool_parser:  # pyright: ignore[reportAny]
                    self.generator.tool_parser = make_mlx_parser(
                        tok.tool_call_start,
                        tok.tool_call_end,
                        tok.tool_parser,  # pyright: ignore[reportAny]
                    )

                self.generator = self.generator.build()

                # Start persistent KV transfer server for disagg decode runners.
                if isinstance(
                    self.instance,
                    (DisaggregatedInstance, TensorPrefillDisaggInstance),
                ):
                    is_decode = (
                        self.bound_instance.bound_node_id
                        == self.instance.decode_node_id
                    )
                    if is_decode:
                        from exo.worker.engines.mlx.kv_transfer import (
                            KV_TRANSFER_PORT,
                            KVTransferServer,
                        )

                        self.kv_transfer_server = KVTransferServer(KV_TRANSFER_PORT)

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

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())
                logger.info("runner ready")

            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case DisaggPrefill(
                task_params=task_params,
                command_id=command_id,
                decode_node_host=decode_host,
                decode_node_port=decode_port,
            ) if isinstance(self.current_status, RunnerReady):
                self.handle_disagg_prefill(
                    task, task_params, command_id, decode_host, decode_port
                )

            case TensorParallelDisaggPrefill(
                task_params=task_params,
                command_id=command_id,
                decode_node_host=decode_host,
                decode_node_port=decode_port,
            ) if isinstance(self.current_status, RunnerReady):
                self.handle_tp_disagg_prefill(
                    task, task_params, command_id, decode_host, decode_port
                )

            case DisaggDecode(
                task_params=task_params,
                command_id=command_id,
                kv_transfer_port=kv_port,
            ) if isinstance(self.current_status, RunnerReady):
                self.handle_disagg_decode(task, task_params, command_id, kv_port)

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
        if self.kv_transfer_server is not None:  # pyright: ignore[reportAny]
            self.kv_transfer_server.close()  # pyright: ignore[reportAny]
            self.kv_transfer_server = None
        if isinstance(self.generator, InferenceGenerator):
            self.generator.close()
        mx.clear_cache()
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
                del self.active_tasks[task_id]

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
        self, response: GenerationResponse | ToolCallResponse, command_id: CommandId
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

    def handle_disagg_prefill(
        self,
        task: DisaggPrefill,
        task_params: TextGenerationTaskParams,
        command_id: CommandId,
        decode_host: str,
        decode_port: int,
    ) -> None:
        logger.info(f"received disaggregated prefill request: {task}")
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)
        assert isinstance(self.generator, InferenceGenerator)
        gen = cast(SequentialGenerator, self.generator)

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
                send_kv_cache_pipelined_sync,
            )
            from exo.worker.engines.mlx.utils_mlx import apply_chat_template

            prompt = apply_chat_template(gen.tokenizer, task_params)
            all_prompt_tokens = encode_prompt(gen.tokenizer, prompt)

            # Template-aware: append the <|channel|> token so the
            # decode node's first generated token is the channel
            # name, not a protocol marker that MoE routing
            # divergence might corrupt.
            from mlx_lm.models.gpt_oss import Model as GptOssModel

            if isinstance(gen.model, GptOssModel):
                all_prompt_tokens = mx.concatenate(
                    [
                        all_prompt_tokens,
                        mx.array([HARMONY_CHANNEL_TOKEN_ID]),
                    ]
                )

            caches = make_kv_cache(model=cast(Model, gen.model))
            sampler = make_sampler(
                temp=task_params.temperature
                if task_params.temperature is not None
                else 0.7,
            )

            # Send more tail tokens so the decode node can
            # re-prefill them with its own model, grounding
            # hidden states for consistent MoE routing.
            reprefill = min(DISAGG_REPREFILL_TOKENS, len(all_prompt_tokens))
            reprefill = max(reprefill, 2)
            last_tokens = all_prompt_tokens[-reprefill:]
            t_pipelined_start = time.monotonic()
            prefill_tps, num_tokens = send_kv_cache_pipelined_sync(
                host=decode_host,
                port=decode_port,
                model=cast(Model, gen.model),
                tokenizer=gen.tokenizer,
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
            gc.collect()
            self._reclaim_gpu_memory()
            self._log_cuda_mem("after-cleanup")

        except Exception as e:
            logger.opt(exception=e).error("DisaggPrefill failed, recovering runner")
            # Best-effort: notify decode node so it fails fast instead
            # of waiting the full 120-second KV receive timeout.
            from exo.worker.engines.mlx.kv_transfer import (
                notify_decode_of_failure,
            )

            notify_decode_of_failure(decode_host, decode_port, str(e))
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=command_id,
                        chunk=ErrorChunk(
                            model=self.model_id,
                            finish_reason="error",
                            error_message=str(e),
                        ),
                    )
                )
            gc.collect()
            self._reclaim_gpu_memory()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerReady())
        logger.info("runner ready")

    def handle_tp_disagg_prefill(
        self,
        task: TensorParallelDisaggPrefill,
        task_params: TextGenerationTaskParams,
        command_id: CommandId,
        decode_host: str,
        decode_port: int,
    ) -> None:
        logger.info(f"received tensor-parallel disagg prefill request: {task}")
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)
        assert isinstance(self.generator, InferenceGenerator)
        gen = cast(SequentialGenerator, self.generator)
        assert gen.group is not None, "TP prefill requires distributed group"

        try:
            from mlx_lm.models.cache import KVCache as PlainKVCache
            from mlx_lm.sample_utils import make_sampler

            from exo.worker.engines.mlx.cache import encode_prompt
            from exo.worker.engines.mlx.constants import (
                DISAGG_REPREFILL_TOKENS,
                HARMONY_CHANNEL_TOKEN_ID,
                KV_BITS,
                KV_GROUP_SIZE,
            )
            from exo.worker.engines.mlx.kv_transfer import (
                gather_sharded_kv_cache,
                send_precomputed_kv_cache_sync,
            )
            from exo.worker.engines.mlx.utils_mlx import apply_chat_template

            prompt = apply_chat_template(gen.tokenizer, task_params)
            all_prompt_tokens = encode_prompt(gen.tokenizer, prompt)

            # Template-aware: same as DisaggPrefill — append
            # <|channel|> so decode starts from the channel name.
            from mlx_lm.models.gpt_oss import Model as GptOssModel

            if isinstance(gen.model, GptOssModel):
                all_prompt_tokens = mx.concatenate(
                    [
                        all_prompt_tokens,
                        mx.array([HARMONY_CHANNEL_TOKEN_ID]),
                    ]
                )

            prompt_tokens = all_prompt_tokens[:-1]
            reprefill = min(DISAGG_REPREFILL_TOKENS, len(all_prompt_tokens))
            reprefill = max(reprefill, 2)
            last_tokens = all_prompt_tokens[-reprefill:]

            caches = [PlainKVCache() for _ in cast(Model, gen.model).layers]
            sampler = make_sampler(
                temp=task_params.temperature
                if task_params.temperature is not None
                else 0.7,
            )

            t_prefill_start = time.monotonic()
            for _ in stream_generate(
                model=cast(Model, gen.model),
                tokenizer=gen.tokenizer,
                prompt=prompt_tokens,
                max_tokens=1,
                sampler=sampler,
                prompt_cache=caches,
                prefill_step_size=4096,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
            ):
                break

            t_prefill_end = time.monotonic()
            prefill_ms = (t_prefill_end - t_prefill_start) * 1000

            for c in caches:
                c.trim(2)

            num_tokens = caches[0].offset
            prefill_tps = num_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
            logger.info(
                f"DISAGG_TIMING tp_prefill_ms={prefill_ms:.1f} "
                f"prefill_tps={prefill_tps:.1f} num_tokens={num_tokens}"
            )

            t_gather_start = time.monotonic()
            gather_sharded_kv_cache(caches, gen.group)
            t_gather_end = time.monotonic()
            logger.info(
                f"DISAGG_TIMING tp_kv_gather_ms={(t_gather_end - t_gather_start) * 1000:.1f}"
            )

            is_kv_sender = (
                isinstance(self.instance, TensorPrefillDisaggInstance)
                and self.bound_instance.bound_node_id == self.instance.kv_sender_node_id
            )
            if is_kv_sender:
                t_send_start = time.monotonic()
                send_precomputed_kv_cache_sync(
                    host=decode_host,
                    port=decode_port,
                    cache=caches,
                    last_tokens=last_tokens,
                )
                t_send_end = time.monotonic()
                logger.info(
                    f"DISAGG_TIMING tp_kv_send_ms={(t_send_end - t_send_start) * 1000:.1f}"
                )
            else:
                # Non-sender rank: free the gathered cache immediately.
                # After all_gather each rank holds the full KV cache
                # but only the sender needs it for the network send.
                # On unified-memory devices (GB10) these arrays consume
                # GPU memory via page cache; releasing them here prevents
                # the non-sender from holding ~2x the KV footprint until
                # the sender finishes its network transfer.
                for c in caches:
                    c.state = (
                        mx.zeros((1, 1, 1, 1)),
                        mx.zeros((1, 1, 1, 1)),
                    )

            # Free KV cache and intermediate tensors to prevent
            # GPU memory accumulation across consecutive requests.
            self._log_cuda_mem("before-cleanup")
            del caches
            gc.collect()
            self._reclaim_gpu_memory()
            self._log_cuda_mem("after-cleanup")

        except Exception as e:
            logger.opt(exception=e).error(
                "TensorParallelDisaggPrefill failed, recovering runner"
            )
            # Best-effort: notify decode node so it fails fast instead
            # of waiting the full 120-second KV receive timeout.
            if self.device_rank == 0:
                from exo.worker.engines.mlx.kv_transfer import (
                    notify_decode_of_failure,
                )

                notify_decode_of_failure(decode_host, decode_port, str(e))
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=command_id,
                        chunk=ErrorChunk(
                            model=self.model_id,
                            finish_reason="error",
                            error_message=str(e),
                        ),
                    )
                )
            gc.collect()
            self._reclaim_gpu_memory()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerReady())
        logger.info("runner ready")

    def handle_disagg_decode(
        self,
        task: DisaggDecode,
        task_params: TextGenerationTaskParams,
        command_id: CommandId,
        kv_port: int,
    ) -> None:
        logger.info(
            f"received disaggregated decode request, waiting for KV cache on port {kv_port}"
        )
        self.update_status(RunnerRunning())
        self.acknowledge_task(task)
        assert isinstance(self.generator, InferenceGenerator)
        gen = cast(SequentialGenerator, self.generator)

        try:
            from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
            from mlx_lm.models.gpt_oss import Model as GptOssModel
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
                parse_deepseek_v32,
                parse_gpt_oss,
                parse_thinking_models,
                wrap_stream_generate,
            )

            t_kv_wait_start = time.monotonic()
            if self.kv_transfer_server is not None:  # pyright: ignore[reportAny]
                received_caches, last_tokens = self.kv_transfer_server.receive()  # pyright: ignore[reportAny]
            else:
                # Fallback for unexpected cases
                received_caches, last_tokens = receive_kv_cache_auto_sync(kv_port)
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
            received_caches = _align_received_caches(gen.model, received_caches)

            sampler = make_sampler(
                temp=task_params.temperature
                if task_params.temperature is not None
                else 0.7,
                top_p=task_params.top_p if task_params.top_p is not None else 1.0,
            )
            max_tokens = task_params.max_output_tokens or MAX_TOKENS

            t_decode_start = time.monotonic()

            # Compute prompt tokens for usage reporting.
            prompt = apply_chat_template(gen.tokenizer, task_params)
            prompt_token_count = len(gen.tokenizer.encode(prompt))

            raw_stream = stream_generate(
                model=gen.model,
                tokenizer=gen.tokenizer,
                prompt=last_tokens,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=received_caches,
                prefill_step_size=DISAGG_REPREFILL_TOKENS,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
            )
            parsed: Generator[
                GenerationResponse | ToolCallResponse | None, None, None
            ] = wrap_stream_generate(
                raw_stream,
                t_decode_start,
                prompt_tokens=prompt_token_count,
            )

            # Template-aware: the prefill node appended
            # <|channel|> to the prompt, so the decode model's
            # first generated token is the channel name.  Inject
            # a matching synthetic <|channel|> into the parser
            # stream so parse_gpt_oss sees a valid Harmony
            # sequence instead of falling back to token
            # stripping.
            if isinstance(gen.model, GptOssModel):

                def _prepend_channel(
                    stream: Generator[GenerationResponse, None, None],
                ) -> Generator[GenerationResponse, None, None]:
                    yield GenerationResponse(
                        text="<|channel|>",
                        token=HARMONY_CHANNEL_TOKEN_ID,
                        finish_reason=None,
                        usage=None,
                    )
                    yield from stream

                parsed = _prepend_channel(parsed)

            if gen.tokenizer.has_thinking:
                parsed = parse_thinking_models(
                    parsed, gen.tokenizer, starts_in_thinking=False
                )
            if isinstance(gen.model, GptOssModel):
                parsed = parse_gpt_oss(parsed)
            elif (
                isinstance(gen.model, DeepseekV32Model)
                and "deepseek" in self.model_id.normalize().lower()
            ):
                parsed = parse_deepseek_v32(parsed)

            tokens_since_last_cancel_check = gen.check_for_cancel_every
            for response in parsed:
                if response is None:
                    continue
                tokens_since_last_cancel_check += 1
                if tokens_since_last_cancel_check >= gen.check_for_cancel_every:
                    tokens_since_last_cancel_check = 0
                    gen._cancelled_tasks.update(self.cancel_receiver.collect())  # pyright: ignore[reportPrivateUsage]
                    want_to_cancel = (task.task_id in gen._cancelled_tasks) or (  # pyright: ignore[reportPrivateUsage]
                        TaskId("CANCEL_CURRENT_TASK") in gen._cancelled_tasks  # pyright: ignore[reportPrivateUsage]
                    )
                    if want_to_cancel:
                        break

                self.send_response(response, command_id)
                if (
                    isinstance(response, GenerationResponse)
                    and response.finish_reason is not None
                ):
                    break

            # Free received KV caches and intermediate tensors.
            del received_caches
            gc.collect()
            mx.clear_cache()

        except Exception as e:
            logger.opt(exception=e).error("DisaggDecode failed, recovering runner")
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=command_id,
                        chunk=ErrorChunk(
                            model=self.model_id,
                            finish_reason="error",
                            error_message=str(e),
                        ),
                    )
                )
            gc.collect()
            mx.clear_cache()
            mx.synchronize()

        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerReady())
        logger.info("runner ready")


@dataclass
class Builder:
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    tool_parser: ToolParser | None = None
    group: mx.distributed.Group | None = None
    kv_prefix_cache: KVPrefixCache | None = None

    def build(
        self,
    ) -> InferenceGenerator:
        assert self.model_id
        assert self.inference_model
        assert self.tokenizer

        return SequentialGenerator(
            model=self.inference_model,
            tokenizer=self.tokenizer,
            group=self.group,
            tool_parser=self.tool_parser,
            kv_prefix_cache=self.kv_prefix_cache,
            model_id=self.model_id,
            device_rank=0 if self.group is None else self.group.rank(),
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
        )
