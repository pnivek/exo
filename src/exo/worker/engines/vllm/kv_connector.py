# pyright: reportAny = false
import contextlib
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import torch
from vllm.config import VllmConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from exo.worker.engines.vllm.disaggregated.adapter import (
    extract_kv_via_slot_mapping,
    to_bf16,
)
from exo.worker.runner.bootstrap import logger

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


# Module-level shared state. Populated by the connector's hooks (running inside
# vLLM's scheduler/worker, same process since V1 multiprocessing is off);
# drained by the producer engine in `serve_prefill` after the request finishes.
#
# `_kv_queue` is the original streaming-connector path ported into this module.
# We defer prefix reuse to vLLM APC and do not keep a separate TorchKVCache.
# 4-tuple: (layer_idx, keys_host_pinned, values_host_pinned, copy_done_event)
# The writer thread does `event.synchronize()` (CPU-side, doesn't block GPU)
# before reading the pinned host bytes.
# 5-tuple: (layer_idx, num_tokens, keys_host_pinned, values_host_pinned, copy_done_event)
# `num_tokens` is the authoritative token count for this item. The writer uses
# it for skip_tokens accounting *and* to slice the keys/values tensors before
# writing to wire — never trusts `keys.shape[0]`, since shape can disagree with
# token count when the source path packs/reshapes (e.g. NVFP4 layouts).
# Bounded: entries hold PINNED host tensors, which on unified-memory systems
# (GB10) come from the same physical pool as GPU memory. An unbounded queue
# with a slow receiver accumulates pinned memory without limit — observed to
# starve NVRM and wedge the host. Bounded puts apply backpressure to the
# forward pass instead; a stuck writer trips the put timeout and aborts the
# request rather than deadlocking or OOMing the machine.
KV_QUEUE_MAX_CHUNKS = 64
KV_QUEUE_PUT_TIMEOUT_SECONDS = 60.0
_kv_queue: queue.Queue[
    tuple[int, int, torch.Tensor, torch.Tensor, torch.cuda.Event] | None
] = queue.Queue(maxsize=KV_QUEUE_MAX_CHUNKS)
# 3-tuple: (layer_idx, arrays_host_or_gpu, copy_done_event_or_none)
# - From save_kv_layer hybrid path: tensors are GPU, event=None (writer .cpu()s)
# - From GDN capture (after both conv+ssm ready): tensors are pinned host,
#   event is a CUDA event the writer must synchronize on before reading
_arrays_queue: queue.Queue[
    tuple[int, list[torch.Tensor], torch.cuda.Event | None] | None
] = queue.Queue(maxsize=KV_QUEUE_MAX_CHUNKS)
# Per-layer tracking of which layers' GDN states have been shipped via the
# async pipeline. Entries here are excluded from the post-writer fallback drain.
_gdn_shipped: set[int] = set()
# Capture gate: True only while a serve_prefill is actively draining the
# queues. Without it, ordinary generation requests (no writer thread running)
# capture their prompt KV into the staging queues where it accumulates as
# pinned memory — the leak the bounded queues turned into a loud queue.Full.
_capture_active: list[bool] = [False]


def set_capture_active(active: bool) -> None:
    _capture_active[0] = active


# Reusable pinned host buffers. torch.empty(pin_memory=True) is a synchronous
# cudaHostAlloc issued on the COMPUTE thread inside save_kv_layer -- measured
# to stall the forward pass badly (producer rate 2.4x slower than the model's
# own prefill floor at 640 allocs per 16K-token request). Buffers cycle:
# save_kv_layer takes, the sender returns after the bytes hit the socket.
_PINNED_POOL_PER_SHAPE = 16
_pinned_pool: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = {}
_pinned_pool_lock = threading.Lock()
_producer_stats: dict[str, float] = {}


def _pinned_take(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    key = (shape, dtype)
    with _pinned_pool_lock:
        bucket = _pinned_pool.get(key)
        if bucket:
            _producer_stats["pool_hits"] = _producer_stats.get("pool_hits", 0) + 1
            return bucket.pop()
    _producer_stats["pool_misses"] = _producer_stats.get("pool_misses", 0) + 1
    return torch.empty(shape, dtype=dtype, pin_memory=True)


def return_pinned_buffers(*tensors: torch.Tensor) -> None:
    # Sender-side: recycle pinned staging buffers once their bytes are on the
    # wire (the kernel has copied them into the socket buffer by then).
    with _pinned_pool_lock:
        for t in tensors:
            key = (tuple(t.shape), t.dtype)
            bucket = _pinned_pool.setdefault(key, [])
            if len(bucket) < _PINNED_POOL_PER_SHAPE:
                bucket.append(t)


def get_producer_stats() -> dict[str, float]:
    return _producer_stats
_captured_layers: dict[int, dict[str, torch.Tensor]] = {}
_captured_arrays: dict[int, list[torch.Tensor]] = {}
# Hybrid-model SSM/conv state captured via causal_conv1d + delta-rule patches.
# Values are tensors under "conv"/"ssm"; the conv patch also stores the plain
# int cache index under "ci", so the value type stays loose.
_gdn_states: dict[int, dict[str, Any]] = {}
_gdn_layer_order: list[int] = []
_gdn_call_idx: list[int] = [0]
_ssm_call_idx: list[int] = [0]
# Per-layer save_kv_layer call diagnostics: list of slot_mapping sizes seen.
_save_kv_layer_diag: dict[int, list[int]] = {}
# Side CUDA stream for K/V extract + async D2H, so vLLM's compute stream
# isn't blocked on D2H/extract during forward.
_save_stream: torch.cuda.Stream | None = None
# Holds a reference to the set tracked by patched_schedule so
# `reset_capture_state` can clear it between requests.
_apc_extracted_set_ref: dict[str, set[str]] = {}
# request_id → actual APC hit token count (captured at the moment vLLM's
# kv_cache_manager.get_computed_blocks runs, before scheduler chunks the
# remaining tokens). Used by patched_schedule to pre-extract exactly the
# matched portion, not the matched+about-to-forward portion.
_apc_hit_tokens: dict[str, int] = {}


def _get_save_stream() -> torch.cuda.Stream:
    global _save_stream
    if _save_stream is None:
        _save_stream = torch.cuda.Stream()
    return _save_stream


def get_kv_queue() -> queue.Queue[
    tuple[int, int, torch.Tensor, torch.Tensor, torch.cuda.Event] | None
]:
    return _kv_queue


def get_arrays_queue() -> queue.Queue[
    tuple[int, list[torch.Tensor], torch.cuda.Event | None] | None
]:
    return _arrays_queue


def get_gdn_states() -> dict[int, dict[str, Any]]:
    return _gdn_states


def get_gdn_shipped() -> set[int]:
    return _gdn_shipped


def _try_ship_gdn(layer_idx: int) -> None:
    """If both conv and ssm have been captured for `layer_idx`, kick off an
    async pinned D2H on the side stream and enqueue an arrays-state item so
    the writer thread can ship the bytes during forward instead of after.

    Called from BOTH the conv and ssm capture patches. Conv always fires
    before ssm in a Mamba layer's forward, so this is a no-op after conv
    (state lacks ssm) and ships once after ssm. For chunked prefill the
    pair fires once per chunk: we ship every time, and the consumer's
    `arrays[layer_idx] = ...` last-write-wins keeps the final-chunk state
    (Mamba state is cumulative, only the final state matters).
    """
    if not _capture_active[0]:
        return
    state = _gdn_states.get(layer_idx)
    if state is None or "conv" not in state or "ssm" not in state:
        return
    conv_gpu = state["conv"]
    ssm_gpu = state["ssm"]
    side_stream = _get_save_stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        conv_host = torch.empty(conv_gpu.shape, dtype=conv_gpu.dtype, pin_memory=True)
        ssm_host = torch.empty(ssm_gpu.shape, dtype=ssm_gpu.dtype, pin_memory=True)
        conv_host.copy_(conv_gpu, non_blocking=True)
        ssm_host.copy_(ssm_gpu, non_blocking=True)
    event = torch.cuda.Event()
    event.record(side_stream)
    _arrays_queue.put(
        (layer_idx, [conv_host, ssm_host], event),
        timeout=KV_QUEUE_PUT_TIMEOUT_SECONDS,
    )
    _gdn_shipped.add(layer_idx)


def get_save_kv_layer_diag() -> dict[int, list[int]]:
    return _save_kv_layer_diag


def get_captured_layers() -> dict[int, dict[str, torch.Tensor]]:
    return _captured_layers


def get_captured_arrays() -> dict[int, list[torch.Tensor]]:
    return _captured_arrays


def reset_capture_state() -> None:
    while not _kv_queue.empty():
        try:
            _kv_queue.get_nowait()
        except queue.Empty:
            break
    while not _arrays_queue.empty():
        try:
            _arrays_queue.get_nowait()
        except queue.Empty:
            break
    _captured_layers.clear()
    _captured_arrays.clear()
    _gdn_states.clear()
    _gdn_shipped.clear()
    _gdn_call_idx[0] = 0
    _ssm_call_idx[0] = 0
    _save_kv_layer_diag.clear()
    _apc_hit_tokens.clear()
    _producer_stats.clear()
    apc_set = _apc_extracted_set_ref.get("set")
    if apc_set is not None:
        apc_set.clear()


@dataclass
class StreamingConnectorMetadata(KVConnectorMetadata):
    pass


@dataclass
class BatchConnectorMetadata(KVConnectorMetadata):
    pass


class StreamingConnector(KVConnectorBase_V1, SupportsHMA):
    """Original streaming producer connector, kept under the new server abstraction."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._save_count = 0

    # =========================================================================
    # Worker-side hooks (the only ones we actually use)
    # =========================================================================

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if not _capture_active[0]:
            return
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        m = _LAYER_RE.search(layer_name)
        layer_idx_for_diag = int(m.group(1)) if m else -1
        slot_size = int(slot_mapping.shape[0]) if slot_mapping is not None else -1
        is_list_kv = isinstance(kv_layer, (list, tuple))
        # Tag list/tuple as negative so the diag log distinguishes hybrid from
        # non-hybrid even when slot_size is the same.
        _save_kv_layer_diag.setdefault(layer_idx_for_diag, []).append(
            -slot_size if is_list_kv else slot_size
        )

        # Skip decode-step saves (small slot mapping); we only want prefill.
        if slot_mapping is not None and slot_mapping.shape[0] <= 100:
            return
        if m is None:
            return
        layer_idx = int(m.group(1))

        # Hybrid (Mamba+attention) layers: kv_layer is a list/tuple of state
        # tensors (conv + ssm). Send them straight to the arrays queue —
        # they don't live in the paged KV cache. Stay on GPU; the writer
        # thread does the D2H copy via `tensor_to_wire_bytes`.
        if isinstance(kv_layer, (list, tuple)):
            arrays = [
                to_bf16(t)
                for t in cast(list[torch.Tensor] | tuple[torch.Tensor, ...], kv_layer)
            ]
            _arrays_queue.put(
                (layer_idx, arrays, None), timeout=KV_QUEUE_PUT_TIMEOUT_SECONDS
            )
            return

        # Standard attention layers (full or sliding-window): extract K/V
        # via slot_mapping, which points to where vLLM is *writing* this
        # forward step's tokens. Capturing here, before sliding-window
        # eviction in the block pool, is the only way to ship every prompt
        # token's K/V regardless of attention type.
        #
        # All of this work — gather + bf16 cast + D2H — runs on a side
        # CUDA stream into pinned host memory. vLLM's compute stream is
        # never blocked: it only has to record-event for our side stream
        # to wait on, then it continues into the next layer's forward.
        # The writer thread later waits on the CUDA event (CPU-side wait,
        # doesn't block GPU) and ships the already-on-host bytes.
        if slot_mapping is not None:
            t_capture = time.perf_counter()
            try:
                save_stream = _get_save_stream()
                save_stream.wait_stream(torch.cuda.current_stream())
                num_actual = getattr(attn_metadata, "num_actual_tokens", None)
                if num_actual is not None:
                    num_actual = int(num_actual)
                    if num_actual <= 0 or num_actual > int(slot_mapping.shape[0]):
                        num_actual = None
                with torch.cuda.stream(save_stream):
                    keys_gpu, values_gpu = extract_kv_via_slot_mapping(
                        kv_layer, slot_mapping, num_actual_tokens=num_actual
                    )
                    keys_host = _pinned_take(tuple(keys_gpu.shape), keys_gpu.dtype)
                    values_host = _pinned_take(
                        tuple(values_gpu.shape), values_gpu.dtype
                    )
                    keys_host.copy_(keys_gpu, non_blocking=True)
                    values_host.copy_(values_gpu, non_blocking=True)
                    num_tokens = int(keys_gpu.shape[0])
                event = torch.cuda.Event()
                event.record(save_stream)
                _producer_stats["capture_secs"] = _producer_stats.get(
                    "capture_secs", 0.0
                ) + (time.perf_counter() - t_capture)
            except Exception as exc:
                logger.warning(
                    f"save_kv_layer extract failed layer={layer_idx} "
                    f"kv_layer.shape={getattr(kv_layer, 'shape', None)} "
                    f"slot_mapping.shape={slot_mapping.shape}: {exc!r}"
                )
                return
            _kv_queue.put(
                (layer_idx, num_tokens, keys_host, values_host, event),
                timeout=KV_QUEUE_PUT_TIMEOUT_SECONDS,
            )

    def wait_for_save(self) -> None:
        return

    # =========================================================================
    # Scheduler-side hooks (no-ops; we don't load and don't track allocs)
    # =========================================================================

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(self, scheduler_output: Any) -> StreamingConnectorMetadata:
        return StreamingConnectorMetadata()

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self, request: Any, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


class BatchConnector(KVConnectorBase_V1, SupportsHMA):
    """Original batch producer connector, ported for parity with the old branch."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if not _capture_active[0]:
            return
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is not None and slot_mapping.shape[0] <= 100:
            return

        m = _LAYER_RE.search(layer_name)
        if m is None:
            return
        layer_idx = int(m.group(1))

        if isinstance(kv_layer, (list, tuple)):
            _captured_arrays[layer_idx] = [
                to_bf16(t).cpu()
                for t in cast(list[torch.Tensor] | tuple[torch.Tensor, ...], kv_layer)
            ]
            return

        if slot_mapping is None:
            return
        keys, values = extract_kv_via_slot_mapping(kv_layer, slot_mapping)
        prev = _captured_layers.get(layer_idx)
        if prev is None:
            _captured_layers[layer_idx] = {"keys": keys, "values": values}
        else:
            _captured_layers[layer_idx] = {
                "keys": torch.cat([prev["keys"], keys], dim=0),
                "values": torch.cat([prev["values"], values], dim=0),
            }

    def wait_for_save(self) -> None:
        return

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self, request: Any, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(self, scheduler_output: Any) -> BatchConnectorMetadata:
        return BatchConnectorMetadata()


ExoKVProducerConnector = StreamingConnector


# =============================================================================
# Bypass patches — necessary to make our connector usable inside vLLM 1.x.
# Ported from the original branch's prefill_server.py:_patch_vllm_for_connector.
# =============================================================================

_connector_patched = False


def _patch_vllm_for_connector(  # pyright: ignore[reportUnusedFunction]
    connector_class: type[Any],
) -> None:
    """Three patches that make a custom save-only connector cooperate with vLLM.

    1. Suppress `unify_hybrid_kv_cache_specs` ValueError on hybrid (Mamba +
       attention) models — the unifier complains about mixed cache specs we
       don't need to actually unify for save-only operation.
    2. Override `Scheduler._connector_finished` to short-circuit the
       async-save state machine. We're synchronous on the producer side.
    3. Make `KVConnectorFactory._get_connector_class_with_compat` recognize
       our class name and return our class directly, bypassing vLLM's
       registry of built-in connectors.
    """
    global _connector_patched
    if _connector_patched:
        return
    _connector_patched = True

    from vllm.v1.core import kv_cache_utils

    original_unify = kv_cache_utils.unify_hybrid_kv_cache_specs

    def patched_unify(kv_cache_spec: Any) -> None:
        with contextlib.suppress(ValueError):
            original_unify(kv_cache_spec)

    kv_cache_utils.unify_hybrid_kv_cache_specs = patched_unify

    from vllm.v1.core.sched import scheduler as sched_mod

    def patched_connector_finished(
        self: sched_mod.Scheduler, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    sched_mod.Scheduler._connector_finished = patched_connector_finished

    from vllm.distributed.kv_transfer.kv_connector import factory

    if hasattr(factory.KVConnectorFactory, "register_connector"):
        # vLLM 0.21+ has a sanctioned registration API — no override needed.
        for cls in (connector_class, BatchConnector):
            factory.KVConnectorFactory.register_connector(
                cls.__name__, cls.__module__, cls.__name__
            )
        if connector_class.__name__ != "ExoKVProducerConnector":
            factory.KVConnectorFactory.register_connector(
                "ExoKVProducerConnector",
                connector_class.__module__,
                connector_class.__name__,
            )
    else:
        # vLLM 0.19-era: intercept the private resolver.
        original_get = factory.KVConnectorFactory._get_connector_class_with_compat

        def patched_get(kv_transfer_config: KVTransferConfig) -> tuple[Any, Any]:
            kv_conn = kv_transfer_config.kv_connector or ""
            kv_conn_lower = kv_conn.lower()
            if (
                kv_conn
                in {
                    connector_class.__name__,
                    f"{connector_class.__module__}:{connector_class.__name__}",
                    "ExoKVProducerConnector",
                    f"{__name__}:ExoKVProducerConnector",
                    "StreamingConnector",
                    f"{__name__}:StreamingConnector",
                }
                or "streaming_connector" in kv_conn_lower
            ):
                return connector_class, None
            if "batch_connector" in kv_conn_lower:
                return BatchConnector, None
            return original_get(kv_transfer_config)

        factory.KVConnectorFactory._get_connector_class_with_compat = patched_get

    # Patch KVCacheManager.get_computed_blocks so we capture the actual APC-hit
    # token count for each request at the moment vLLM looks it up — *before*
    # the scheduler bumps `req.num_computed_tokens` with the chunked-prefill
    # first-chunk size. Reading `req.num_computed_tokens` post-schedule yields
    # `apc_hit + first_chunk` and would cause us to extract bytes from blocks
    # that haven't been written yet for the first-chunk tail.
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except ImportError:
        KVCacheManager = None  # noqa: N806

    if KVCacheManager is not None:
        original_get_computed_blocks = KVCacheManager.get_computed_blocks

        def patched_get_computed_blocks(self: Any, request: Any) -> Any:
            result: Any = original_get_computed_blocks(self, request)
            try:
                req_id = getattr(request, "request_id", None)
                if req_id is not None:
                    if isinstance(result, tuple) and len(result) >= 2:
                        num = int(result[1])
                    else:
                        num = 0
                    total = int(getattr(request, "num_tokens", 0) or 0)
                    logger.info(
                        f"APC get_computed_blocks: req={req_id} hit={num} total={total}"
                    )
                    if num > 0:
                        _apc_hit_tokens[req_id] = num
            except Exception:
                logger.opt(exception=True).warning(
                    "patched_get_computed_blocks: capture failed"
                )
            return result

        KVCacheManager.get_computed_blocks = patched_get_computed_blocks

    # Patch Scheduler.schedule so APC-cached prefix blocks are extracted out
    # of the paged pool and pushed to _kv_queue at scheduling time — BEFORE
    # forward runs. Forward will only execute the suffix (vLLM's own APC
    # behavior). save_kv_layer fires for the suffix as usual. The writer
    # thread sees: prefix items from this hook + suffix items from save_kv_layer
    # and ships them in arrival order (prefix before suffix per layer).
    original_schedule = sched_mod.Scheduler.schedule
    _scheduled_apc_extracted: set[str] = set()

    def patched_schedule(self: sched_mod.Scheduler) -> Any:
        if not _capture_active[0]:
            return original_schedule(self)
        scheduler_output = original_schedule(self)
        try:
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or []
            if not new_reqs:
                return scheduler_output
            from exo.worker.engines.vllm.disaggregated.adapter import (
                build_layer_to_group,
                gather_layer_kv_from_blocks,
            )
            from exo.worker.engines.vllm.growable_cache import get_model_runner

            mr = get_model_runner()
            if mr is None:
                return scheduler_output
            cfg = getattr(mr, "_growable_kv_cache_config", None)
            if cfg is None:
                return scheduler_output
            layer_to_group = build_layer_to_group(cfg)
            n_layers = len(mr.kv_caches)

            for new_req in new_reqs:
                req_id = getattr(new_req, "req_id", None)
                if req_id is None or req_id in _scheduled_apc_extracted:
                    continue
                pre_layers_shipped = 0
                pre_bytes_shipped = 0
                req = self.requests.get(req_id)
                if req is None:
                    continue
                # Use the count captured by patched_get_computed_blocks (the
                # actual APC hit), NOT req.num_computed_tokens — that field has
                # already been bumped by the scheduler with the first chunk's
                # about-to-forward token count and would over-extract.
                num_apc = _apc_hit_tokens.get(req_id, 0)
                req_total = int(getattr(req, "num_tokens", 0) or 0)
                req_computed = int(getattr(req, "num_computed_tokens", 0) or 0)
                logger.info(
                    f"APC patched_schedule: req={req_id} apc_hit={num_apc} "
                    f"req.num_computed_tokens={req_computed} req.num_tokens={req_total}"
                )
                if num_apc <= 0:
                    _scheduled_apc_extracted.add(req_id)
                    continue
                # Pull the request's full per-group block list from
                # scheduler_output.scheduled_new_reqs[i].block_ids — that field
                # includes APC-cached prefix blocks. The KVCacheManager's
                # `req_to_blocks` only tracks newly-allocated blocks for this
                # step's suffix, so reading from there misses the prefix and
                # makes gather return ~bock_count_suffix tokens of garbage.
                req_block_ids_per_group: tuple[list[int], ...] | None = getattr(
                    new_req, "block_ids", None
                )
                if not req_block_ids_per_group:
                    logger.warning(
                        f"APC pre-extract: new_req.block_ids missing for {req_id}"
                    )
                    _scheduled_apc_extracted.add(req_id)
                    continue
                save_stream = _get_save_stream()
                save_stream.wait_stream(torch.cuda.current_stream())
                first_log_done = False
                # Run the entire gather + cast + pinned alloc + D2H on the
                # side stream — scheduler thread only issues kernel launches
                # and records an event per layer. Compute stream is untouched.
                with torch.cuda.stream(save_stream):
                    for layer_idx in range(n_layers):
                        kv_layer = mr.kv_caches[layer_idx]
                        if isinstance(kv_layer, (list, tuple)):
                            continue
                        gi = (
                            layer_to_group[layer_idx]
                            if layer_idx < len(layer_to_group)
                            else 0
                        )
                        if gi >= len(req_block_ids_per_group):
                            continue
                        block_ids = list(req_block_ids_per_group[gi])
                        if not block_ids:
                            continue
                        keys_gpu, values_gpu = gather_layer_kv_from_blocks(
                            kv_layer, block_ids, num_apc
                        )
                        if not first_log_done:
                            first_log_done = True
                            logger.info(
                                f"APC pre-extract layer={layer_idx}: "
                                f"kv_layer.shape={tuple(kv_layer.shape)} "
                                f"kv_layer.dtype={kv_layer.dtype} "
                                f"len(block_ids)={len(block_ids)} num_apc={num_apc} "
                                f"keys_gpu.shape={tuple(keys_gpu.shape)} "
                                f"keys_gpu.dtype={keys_gpu.dtype}"
                            )
                        if keys_gpu.numel() == 0:
                            continue
                        keys_host = torch.empty(
                            keys_gpu.shape,
                            dtype=keys_gpu.dtype,
                            pin_memory=True,
                        )
                        values_host = torch.empty(
                            values_gpu.shape,
                            dtype=values_gpu.dtype,
                            pin_memory=True,
                        )
                        keys_host.copy_(keys_gpu, non_blocking=True)
                        values_host.copy_(values_gpu, non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(save_stream)
                        _kv_queue.put(
                            (layer_idx, num_apc, keys_host, values_host, event),
                            timeout=KV_QUEUE_PUT_TIMEOUT_SECONDS,
                        )
                        pre_layers_shipped += 1
                        pre_bytes_shipped += (
                            keys_host.numel() * keys_host.element_size()
                            + values_host.numel() * values_host.element_size()
                        )
                logger.info(
                    f"APC pre-extract done: req={req_id} layers={pre_layers_shipped} "
                    f"tokens={num_apc} bytes={pre_bytes_shipped}"
                )
                _scheduled_apc_extracted.add(req_id)
        except Exception:
            logger.opt(exception=True).warning(
                "patched_schedule: APC pre-extract failed; continuing"
            )
        return scheduler_output

    sched_mod.Scheduler.schedule = patched_schedule
    # Reset the per-request-extracted set when reset_capture_state runs.
    _apc_extracted_set_ref["set"] = _scheduled_apc_extracted
    logger.info("Installed vLLM connector bypass patches")


# =============================================================================
# Hybrid-model GDN state capture (Qwen3.5/3.6 etc.).
# Patches the conv1d kernel + delta-rule fns to grab conv/ssm states per layer.
# =============================================================================

_gdn_patched = False


def _patch_gdn_capture() -> None:  # pyright: ignore[reportUnusedFunction]
    global _gdn_patched
    if _gdn_patched:
        return
    _gdn_patched = True

    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as cc_mod
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn as orig_fn,
        )
    except ImportError:
        return

    def patched_fn(
        *args: Any, conv_states: Any = None, cache_indices: Any = None, **kwargs: Any
    ) -> Any:
        result = orig_fn(
            *args, conv_states=conv_states, cache_indices=cache_indices, **kwargs
        )
        if conv_states is not None and cache_indices is not None:
            x = args[0] if args else None
            if x is not None and x.shape[0] <= 100:
                return result
            ci: int = cache_indices[0].item() if cache_indices.numel() > 0 else 0
            idx = _gdn_call_idx[0]
            if _gdn_layer_order and idx < len(_gdn_layer_order) * 100:
                layer_idx = _gdn_layer_order[idx % len(_gdn_layer_order)]
                # `.contiguous()` decouples the slice from the underlying
                # buffer; D2H is deferred to the writer thread.
                conv_at_ci = conv_states[ci : ci + 1].transpose(-1, -2).contiguous()
                _gdn_states.setdefault(layer_idx, {})["conv"] = conv_at_ci
                _gdn_states[layer_idx]["ci"] = ci
                # Don't ship from here: conv fires before ssm in a Mamba
                # forward, so state["ssm"] is either missing (chunk 1) or
                # stale from the previous chunk (chunk N>=2). Shipping here
                # would emit a mismatched (conv_N, ssm_{N-1}) pair that the
                # ssm patch's later ship would overwrite. Just wait for ssm.
            _gdn_call_idx[0] += 1
        return result

    cc_mod.causal_conv1d_fn = patched_fn
    import sys

    for mod in list(sys.modules.values()):
        if mod is cc_mod:
            continue
        # transformers' image_processing_* shims have a lazy __getattr__
        # that emits a noisy deprecation warning on every attribute probe.
        # They never use causal_conv1d_fn, so skip them.
        mod_name = getattr(mod, "__name__", "") or ""
        if mod_name.startswith("transformers."):
            continue
        if (
            mod.__dict__.get("causal_conv1d_fn") is orig_fn
            if hasattr(mod, "__dict__")
            else False
        ):
            cast(Any, mod).causal_conv1d_fn = patched_fn
    logger.info("Patched causal_conv1d_fn for GDN conv-state capture")

    # The GDN delta-rule functions live in `mamba/gdn_linear_attn` (defined or
    # re-imported there) and may also be re-exported by model modules. Patch
    # all candidate modules + propagate to anywhere they're imported.
    candidate_modules = [
        "vllm.model_executor.layers.mamba.gdn_linear_attn",
        "vllm.model_executor.models.qwen3_next",
        "vllm.model_executor.models.qwen3_5",
    ]
    fn_names = ("fi_chunk_gated_delta_rule", "fla_chunk_gated_delta_rule")
    patched_targets: list[str] = []

    for mod_path in candidate_modules:
        try:
            mod = __import__(mod_path, fromlist=["*"])
        except ImportError:
            continue
        for fn_name in fn_names:
            orig = getattr(mod, fn_name, None)
            if orig is None:
                continue

            def make_patched(orig_fn_inner: Any) -> Any:
                def patched_chunk(*args: Any, **kwargs: Any) -> Any:
                    result = orig_fn_inner(*args, **kwargs)
                    output_final_state = kwargs.get("output_final_state", False)
                    if (
                        output_final_state
                        and isinstance(result, tuple)
                        and len(result) == 2
                    ):
                        _, ssm_state = result
                        idx = _ssm_call_idx[0]
                        if _gdn_layer_order and idx < len(_gdn_layer_order) * 100:
                            layer_idx = _gdn_layer_order[idx % len(_gdn_layer_order)]
                            _gdn_states.setdefault(layer_idx, {})["ssm"] = ssm_state
                            _try_ship_gdn(layer_idx)
                        _ssm_call_idx[0] += 1
                    return result

                return patched_chunk

            patched_fn = make_patched(orig)
            setattr(mod, fn_name, patched_fn)
            patched_targets.append(f"{mod_path}.{fn_name}")
            # Propagate to any module that imported the original function.
            import sys as _sys

            for other in list(_sys.modules.values()):
                if other is mod:
                    continue
                other_name = getattr(other, "__name__", "") or ""
                # Skip transformers — see causal_conv1d_fn loop above.
                if other_name.startswith("transformers."):
                    continue
                if other.__dict__.get(fn_name) is orig:
                    setattr(other, fn_name, patched_fn)
                    patched_targets.append(f"{other.__name__}.{fn_name} (propagated)")
    if patched_targets:
        logger.info(f"Patched delta-rule fns for SSM capture: {patched_targets}")
    else:
        logger.warning(
            "GDN SSM-capture patch installed no targets — hybrid models may miss ssm state"
        )


def init_gdn_layer_order(kv_caches: Any) -> None:
    """Identify hybrid layers (those with list/tuple kv_cache entries)."""
    _gdn_layer_order.clear()
    for li in range(len(kv_caches)):
        kv = kv_caches[li]
        if isinstance(kv, (list, tuple)) and len(kv) > 1:
            _gdn_layer_order.append(li)
    if _gdn_layer_order:
        logger.info(f"GDN layer order: {len(_gdn_layer_order)} hybrid layers detected")
