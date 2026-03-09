"""KV cache serialization and TCP transfer for disaggregated inference.

Handles serializing KV cache arrays from the prefill node and sending them
over a TCP socket to the decode node, which deserializes and reconstructs
the cache for token generation.

Supports two protocols:
- Bulk: serialize entire KV cache as NPZ, send as single blob (legacy)
- Pipelined (KVPS): stream KV deltas as raw frames during prefill, overlapping
  network transfer with GPU compute for lower TTFT
"""

import asyncio
import contextlib
import queue
import socket
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

import mlx.core as mx
import numpy as np
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.mlx import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

# Port for direct KV cache transfer between nodes.
KV_TRANSFER_PORT = 52416

# Pipelined protocol constants
_MAGIC = b"KVPS"
_VERSION = 0x02
_FRAME_CHUNK = 0x01
_FRAME_LAST_TOKENS = 0x02
_FRAME_LAYER = 0x03
_FRAME_ERROR = 0xFE
_FRAME_END = 0xFF

# Dtype flags for wire protocol
_DTYPE_FLOAT16 = 0x00
_DTYPE_BFLOAT16 = 0x01


def _mlx_to_numpy(arr: mx.array) -> "np.ndarray[Any, Any]":
    """Convert an MLX array to numpy, handling bfloat16 which numpy doesn't support.

    For bfloat16 arrays we bitcast to uint16 (same 2 bytes, zero precision loss)
    rather than converting to float16 (lossy — incompatible exponent range).
    The receiver counterpart uses np.frombuffer(data, dtype=np.uint16) and then
    mx.array(...).view(mx.bfloat16) to recover the exact original values.
    """
    if arr.dtype == mx.bfloat16:
        return np.array(arr.view(mx.uint16), copy=False)
    return np.array(arr, copy=False)


# ---------------------------------------------------------------------------
# Bulk protocol (legacy, kept for backward compatibility)
# ---------------------------------------------------------------------------


def serialize_kv_cache(cache: KVCacheType, last_tokens: mx.array) -> bytes:
    """Serialize a KV cache and last tokens to bytes for network transfer.

    Uses numpy npz format internally for reliable cross-platform serialization.
    Each layer's keys and values are saved as numpy arrays along with offsets.
    bfloat16 arrays are converted to float16 since numpy lacks bfloat16 support.
    """
    arrays: dict[str, "np.ndarray[Any, Any]"] = {}
    # Track if original dtype was bfloat16 so deserializer can convert back
    uses_bfloat16 = len(cache) > 0 and cache[0].keys.dtype == mx.bfloat16

    for i, c in enumerate(cache):
        keys_np: np.ndarray[Any, Any] = _mlx_to_numpy(c.keys)
        values_np: np.ndarray[Any, Any] = _mlx_to_numpy(c.values)
        arrays[f"layer_{i}_keys"] = keys_np
        arrays[f"layer_{i}_values"] = values_np
        arrays[f"layer_{i}_offset"] = np.array([c.offset], dtype=np.int64)

    arrays["last_tokens"] = np.array(last_tokens, copy=False)
    arrays["num_layers"] = np.array([len(cache)], dtype=np.int64)
    arrays["uses_bfloat16"] = np.array([1 if uses_bfloat16 else 0], dtype=np.int64)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f, **arrays)  # pyright: ignore[reportArgumentType]
        tmp_path = Path(f.name)

    data = tmp_path.read_bytes()
    tmp_path.unlink()

    logger.info(
        f"Serialized KV cache: {len(cache)} layers, {len(data) / 1024 / 1024:.1f} MB"
    )
    return data


def deserialize_kv_cache(data: bytes) -> tuple[list[KVCache], mx.array]:
    """Deserialize bytes back into a KV cache and last tokens.

    Returns a list of KVCache objects with keys/values restored and the
    last tokens array needed to seed decode.
    """
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        f.write(data)
        tmp_path = Path(f.name)

    loaded: Any = np.load(str(tmp_path), allow_pickle=False)  # pyright: ignore[reportAny]
    tmp_path.unlink()

    num_layers = int(loaded["num_layers"][0])  # pyright: ignore[reportAny]
    last_tokens = mx.array(loaded["last_tokens"])  # pyright: ignore[reportAny]
    uses_bfloat16 = (
        bool(loaded["uses_bfloat16"][0]) if "uses_bfloat16" in loaded else False  # pyright: ignore[reportAny]
    )

    caches: list[KVCache] = []
    for i in range(num_layers):
        cache = KVCache()
        keys = mx.array(loaded[f"layer_{i}_keys"])  # pyright: ignore[reportAny]
        values = mx.array(loaded[f"layer_{i}_values"])  # pyright: ignore[reportAny]
        if uses_bfloat16:
            # Bitcast uint16 bits back to bfloat16 — lossless recovery
            # matching the serialize path which uses view(mx.uint16).
            keys = keys.view(mx.bfloat16)
            values = values.view(mx.bfloat16)
        cache.state = (keys, values)
        cache.offset = int(loaded[f"layer_{i}_offset"][0])  # pyright: ignore[reportAny]
        caches.append(cache)

    loaded.close()  # pyright: ignore[reportAny]

    logger.info(
        f"Deserialized KV cache: {num_layers} layers, {len(data) / 1024 / 1024:.1f} MB"
    )
    return caches, last_tokens


async def send_kv_cache(
    host: str,
    port: int,
    cache: KVCacheType,
    last_tokens: mx.array,
    retries: int = 30,
    delay: float = 1.0,
) -> None:
    """Serialize and send KV cache to a remote decode node over TCP.

    Retries connection if the decode node's server isn't ready yet.
    """
    t_ser_start = time.monotonic()
    data = serialize_kv_cache(cache, last_tokens)
    t_ser_end = time.monotonic()
    kv_size_mb = len(data) / 1024 / 1024
    logger.info(
        f"DISAGG_TIMING kv_serialize_ms={(t_ser_end - t_ser_start) * 1000:.1f} "
        f"kv_size_mb={kv_size_mb:.2f}"
    )
    header = struct.pack("!Q", len(data))

    for attempt in range(retries):
        try:
            t_net_start = time.monotonic()
            _, writer = await asyncio.open_connection(host, port)
            writer.write(header)
            writer.write(data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            t_net_end = time.monotonic()
            net_ms = (t_net_end - t_net_start) * 1000
            logger.info(
                f"DISAGG_TIMING kv_network_send_ms={net_ms:.1f} "
                f"bandwidth_mbps={kv_size_mb / (net_ms / 1000) * 8:.1f}"
            )
            return
        except (ConnectionRefusedError, OSError) as e:
            if attempt < retries - 1:
                logger.debug(
                    f"Connection to {host}:{port} failed (attempt {attempt + 1}/{retries}): {e}"
                )
                await asyncio.sleep(delay)
            else:
                raise ConnectionError(
                    f"Failed to connect to decode node at {host}:{port} after {retries} attempts"
                ) from e


async def receive_kv_cache(port: int) -> tuple[list[KVCache], mx.array]:
    """Start a TCP server and wait for one KV cache transfer.

    Returns the deserialized KV cache and last tokens once received.
    """
    result_queue: asyncio.Queue[tuple[list[KVCache], mx.array]] = asyncio.Queue()

    async def handle_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        header = await reader.readexactly(8)
        length: int = struct.unpack("!Q", header)[0]  # pyright: ignore[reportAny]
        kv_size_mb = length / 1024 / 1024
        logger.info(f"Receiving KV cache: {kv_size_mb:.1f} MB")

        t_recv_start = time.monotonic()
        data = await reader.readexactly(length)
        t_recv_end = time.monotonic()
        recv_ms = (t_recv_end - t_recv_start) * 1000
        logger.info(
            f"DISAGG_TIMING kv_network_recv_ms={recv_ms:.1f} "
            f"kv_size_mb={kv_size_mb:.2f}"
        )

        t_deser_start = time.monotonic()
        cache, last_tokens = deserialize_kv_cache(data)
        t_deser_end = time.monotonic()
        logger.info(
            f"DISAGG_TIMING kv_deserialize_ms={(t_deser_end - t_deser_start) * 1000:.1f}"
        )

        await result_queue.put((cache, last_tokens))

        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_connection, "0.0.0.0", port)
    logger.info(f"KV cache transfer server listening on port {port}")

    async with server:
        cache, last_tokens = await result_queue.get()

    return cache, last_tokens


def send_kv_cache_sync(
    host: str, port: int, cache: KVCacheType, last_tokens: mx.array
) -> None:
    """Synchronous wrapper for send_kv_cache (for use in runner subprocess)."""
    asyncio.run(send_kv_cache(host, port, cache, last_tokens))


def receive_kv_cache_sync(port: int) -> tuple[list[KVCache], mx.array]:
    """Synchronous wrapper for receive_kv_cache (for use in runner subprocess)."""
    return asyncio.run(receive_kv_cache(port))


# ---------------------------------------------------------------------------
# Pipelined protocol (KVPS) — overlaps KV transfer with prefill compute
# ---------------------------------------------------------------------------


@dataclass
class KVChunkMessage:
    """A chunk of KV cache deltas for a range of tokens across all layers."""

    chunk_index: int
    start_offset: int
    num_tokens: int
    layer_keys: list["np.ndarray[Any, Any]"] = field(repr=False)
    layer_values: list["np.ndarray[Any, Any]"] = field(repr=False)


# Queue item type: KVChunkMessage, (n_tokens, ndarray) for last tokens, or None sentinel
_QueueItem = KVChunkMessage | tuple[int, "np.ndarray[Any, Any]"] | None


def _connect_with_retries(
    host: str,
    port: int,
    retries: int = 30,
    delay: float = 1.0,
    send_timeout: float = 120.0,
) -> socket.socket:
    """Connect a TCP socket with retries.

    The returned socket has a send timeout (``SO_SNDTIMEO``) configured so
    that ``sendall`` will raise ``TimeoutError`` instead of blocking forever
    when the remote side stops reading (e.g. the decode receiver already
    timed out and moved on).
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(10.0)  # connect timeout
            sock.connect((host, port))
            # Set a send timeout so _sendall doesn't block forever when the
            # receiver has closed/abandoned the connection.
            sock.settimeout(send_timeout)
            return sock
        except (ConnectionRefusedError, OSError) as exc:
            sock.close()
            last_exc = exc
            if attempt < retries - 1:
                logger.debug(
                    f"Pipelined connect to {host}:{port} failed "
                    f"(attempt {attempt + 1}/{retries}): {exc}"
                )
                time.sleep(delay)
    raise ConnectionError(
        f"Failed to connect to decode node at {host}:{port} after {retries} attempts"
    ) from last_exc


def notify_decode_of_failure(host: str, port: int, error_message: str) -> None:
    """Best-effort: connect to decode node and send a KVPS error frame.

    If the decode node is waiting on ``KVTransferServer.receive()``, this
    lets it fail immediately instead of waiting the full 120-second timeout.
    Any exception is swallowed — the decode side will fall back to its
    timeout if this notification doesn't get through.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        # Minimal KVPS v2 header so receiver recognises the protocol
        header = _MAGIC + struct.pack("!BIBI", _VERSION, 0, _DTYPE_FLOAT16, 0)
        header += struct.pack("!II", 1, 64)  # dummy n_kv_heads, head_dim
        sock.sendall(header)
        # Error frame: type(1) + msg_len(4) + msg_bytes
        msg_bytes = error_message.encode("utf-8")[:1024]
        sock.sendall(struct.pack("!BI", _FRAME_ERROR, len(msg_bytes)) + msg_bytes)
        sock.close()
        logger.info(f"Notified decode node of prefill failure: {error_message[:80]}")
    except Exception as exc:
        logger.debug(f"Could not notify decode node of failure: {exc}")


def _sendall(sock: socket.socket, data: bytes | memoryview) -> None:
    """Send all bytes, raise on failure (respects socket timeout)."""
    sock.sendall(data)


def _serialize_chunk_frame(chunk: KVChunkMessage) -> bytes:
    """Serialize a KVChunkMessage into a raw KVPS chunk frame."""
    parts: list[bytes] = []
    # Frame header: type(1) + chunk_index(4) + start_offset(4) + num_tokens(4)
    parts.append(
        struct.pack(
            "!BIII",
            _FRAME_CHUNK,
            chunk.chunk_index,
            chunk.start_offset,
            chunk.num_tokens,
        )
    )
    for layer_idx in range(len(chunk.layer_keys)):
        k_bytes = chunk.layer_keys[layer_idx].tobytes()
        v_bytes = chunk.layer_values[layer_idx].tobytes()
        parts.append(struct.pack("!I", len(k_bytes)))
        parts.append(k_bytes)
        parts.append(struct.pack("!I", len(v_bytes)))
        parts.append(v_bytes)
    return b"".join(parts)


def _sender_thread_fn(
    sock: socket.socket,
    send_queue: queue.Queue[_QueueItem],
    error_event: threading.Event,
) -> None:
    """Background thread that drains the queue and writes frames to the socket."""
    try:
        while True:
            item = send_queue.get()
            if item is None:
                # End sentinel — send END frame and exit
                _sendall(sock, struct.pack("!B", _FRAME_END))
                return
            if isinstance(item, tuple):
                # LastTokensMessage: (n_tokens, int32 ndarray)
                n_tokens: int = item[0]
                tokens_np: np.ndarray[Any, Any] = item[1]
                tok_bytes: bytes = tokens_np.astype(np.int32).tobytes()
                _sendall(
                    sock,
                    struct.pack("!BI", _FRAME_LAST_TOKENS, n_tokens) + tok_bytes,
                )
                continue
            # KVChunkMessage
            assert isinstance(item, KVChunkMessage)
            chunk_index = item.chunk_index
            frame_data = _serialize_chunk_frame(item)
            # Release numpy arrays immediately — on unified memory (GB10),
            # these consume GPU memory via page cache.
            item.layer_keys.clear()
            item.layer_values.clear()
            del item
            t_send_start = time.monotonic()
            _sendall(sock, frame_data)
            t_send_end = time.monotonic()
            chunk_mb = len(frame_data) / 1024 / 1024
            del frame_data
            logger.info(
                f"DISAGG_TIMING pipelined_chunk_send_ms={(t_send_end - t_send_start) * 1000:.1f} "
                f"chunk_index={chunk_index} chunk_mb={chunk_mb:.2f}"
            )
    except Exception as exc:
        logger.error(f"Pipelined sender thread error: {exc}")
        error_event.set()


def extract_kv_delta(
    cache: KVCacheType,
    prev_offset: int,
    current_offset: int,
    chunk_index: int,
) -> KVChunkMessage:
    """Extract a KV cache delta between two offsets across all layers.

    Copies the slice to numpy immediately so the MLX buffer can be reused.
    """
    from mlx_lm.models.cache import RotatingKVCache

    num_tokens = current_offset - prev_offset
    layer_keys: list["np.ndarray[Any, Any]"] = []
    layer_values: list["np.ndarray[Any, Any]"] = []
    for c in cache:
        if isinstance(c, RotatingKVCache) and c._idx >= 0:
            # RotatingKVCache wraps its buffer — global offsets don't map to
            # buffer indices.  After _update_concat (batch prefill), _idx ==
            # keys.shape[2] and the buffer is in temporal order.  Map global
            # offsets to buffer coordinates.
            buf_start_token = c.offset - c._idx
            buf_idx_start = max(0, prev_offset - buf_start_token)
            buf_idx_end = min(c._idx, current_offset - buf_start_token)
            if buf_idx_end > buf_idx_start:
                k_slice = c.keys[:, :, buf_idx_start:buf_idx_end, :]
                v_slice = c.values[:, :, buf_idx_start:buf_idx_end, :]
            else:
                k_slice = c.keys[:, :, :0, :]
                v_slice = c.values[:, :, :0, :]
        else:
            # Standard KVCache: buffer indices match global offsets.
            k_slice = c.keys[:, :, prev_offset:current_offset, :]
            v_slice = c.values[:, :, prev_offset:current_offset, :]
        mx.eval(k_slice, v_slice)
        layer_keys.append(_mlx_to_numpy(k_slice).copy())
        layer_values.append(_mlx_to_numpy(v_slice).copy())
    return KVChunkMessage(
        chunk_index=chunk_index,
        start_offset=prev_offset,
        num_tokens=num_tokens,
        layer_keys=layer_keys,
        layer_values=layer_values,
    )


def extract_single_layer_kv(
    c: "KVCache | Any",
    num_tokens: int,
) -> tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
    """Extract KV for one cache layer, all tokens up to num_tokens.

    The cache state must already be materialized (mx.eval'd).
    Returns (keys_np, values_np) as numpy copies ready for network transfer.
    """
    from mlx_lm.models.cache import RotatingKVCache

    if isinstance(c, RotatingKVCache) and c._idx >= 0:
        buf_end: int = min(c._idx, num_tokens)
        k: mx.array = c.keys[:, :, :buf_end, :]
        v: mx.array = c.values[:, :, :buf_end, :]
    else:
        k = c.keys[:, :, :num_tokens, :]
        v = c.values[:, :, :num_tokens, :]
    return _mlx_to_numpy(k).copy(), _mlx_to_numpy(v).copy()


def _serialize_layer_frame(
    layer_idx: int,
    num_tokens: int,
    k_np: "np.ndarray[Any, Any]",
    v_np: "np.ndarray[Any, Any]",
) -> bytes:
    """Serialize a single layer's KV into a _FRAME_LAYER frame.

    Frame layout: [0x03][4B layer_idx][4B num_tokens][K data][V data]
    """
    k_bytes = k_np.tobytes()
    v_bytes = v_np.tobytes()
    header = struct.pack("!BII", _FRAME_LAYER, layer_idx, num_tokens)
    return header + k_bytes + v_bytes


_LayerQueueItem = (
    tuple[int, int, "np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]
    | tuple[int, "np.ndarray[Any, Any]"]
    | None
)


def _layer_sender_thread_fn(
    sock: socket.socket,
    send_queue: "queue.Queue[_LayerQueueItem]",
    error_event: threading.Event,
) -> None:
    """Background thread that drains per-layer KV frames from the queue and writes to socket."""
    try:
        while True:
            item: _LayerQueueItem = send_queue.get()
            if item is None:
                # End sentinel — send END frame and exit
                _sendall(sock, struct.pack("!B", _FRAME_END))
                return
            if len(item) == 2:
                # LastTokensMessage: (n_tokens, int32 ndarray)
                n_tokens = int(item[0])
                tokens_np: np.ndarray[Any, Any] = item[1]
                tok_bytes: bytes = tokens_np.astype(np.int32).tobytes()
                _sendall(
                    sock,
                    struct.pack("!BI", _FRAME_LAST_TOKENS, n_tokens) + tok_bytes,
                )
                continue
            # Per-layer KV: (layer_idx, num_tokens, k_np, v_np)
            layer_idx = int(item[0])
            num_tokens_val = int(item[1])
            k_np: np.ndarray[Any, Any] = item[2]
            v_np: np.ndarray[Any, Any] = item[3]
            frame_data = _serialize_layer_frame(layer_idx, num_tokens_val, k_np, v_np)
            del k_np, v_np
            t_send_start = time.monotonic()
            _sendall(sock, frame_data)
            t_send_end = time.monotonic()
            frame_mb = len(frame_data) / 1024 / 1024
            del frame_data
            logger.info(
                f"DISAGG_TIMING per_layer_send_ms={(t_send_end - t_send_start) * 1000:.1f} "
                f"layer_idx={layer_idx} frame_mb={frame_mb:.2f}"
            )
    except Exception as exc:
        logger.error(f"Per-layer sender thread error: {exc}")
        error_event.set()


_PER_LAYER_PREFILL_STEP_SIZE: int = 8192


def send_kv_cache_per_layer_sync(
    host: str,
    port: int,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt_tokens: mx.array,
    last_tokens: mx.array,
    cache: KVCacheType,
    sampler: Callable[[mx.array], mx.array],
) -> tuple[float, int]:
    """Run per-layer prefill, streaming each layer's KV as soon as it completes.

    Decomposes the forward pass layer-by-layer: process prompt tokens through
    layer 0, immediately start sending layer 0's KV while layer 1 computes.

    Within each layer, tokens are processed in chunks of _PER_LAYER_PREFILL_STEP_SIZE
    to bound attention memory (matching mlx_lm's prefill chunking) and keep CUDA
    graph cache entries bounded (shapes repeat across layers for same chunk sizes).

    Returns (prefill_tps, num_tokens).
    """
    from mlx_lm.models.base import (
        create_attention_mask,  # pyright: ignore[reportUnknownVariableType]
    )
    from mlx_lm.models.gpt_oss import GptOssMoeModel

    from exo.worker.engines.mlx.auto_parallel import set_pipeline_prefill

    num_layers = len(cache)
    num_tokens = len(prompt_tokens)
    kv_send_cap = max(1, num_tokens + 1 - len(last_tokens))
    step = _PER_LAYER_PREFILL_STEP_SIZE

    # Connect to decode node
    sock = _connect_with_retries(host, port)

    # Set up sender thread
    send_queue: queue.Queue[_LayerQueueItem] = queue.Queue(maxsize=4)
    error_event = threading.Event()
    sender = threading.Thread(
        target=_layer_sender_thread_fn,
        args=(sock, send_queue, error_event),
        daemon=True,
    )

    set_pipeline_prefill(model, is_prefill=True)

    t_prefill_start = time.monotonic()

    # --- Per-layer prefill with chunking ---
    # Access model internals to iterate layer-by-layer.
    inner_model = cast(GptOssMoeModel, model.model)

    # Embed tokens and materialize (needed before slicing into chunks)
    x = inner_model.embed_tokens(prompt_tokens[None])
    mx.eval(x)

    # Determine dtype and send KVPS header after first layer populates cache
    header_sent = False

    layer_types: list[str] = cast(list[str], inner_model.layer_types)

    # Deferred extraction state: after computing layer N, we defer its KV
    # extraction until layer N+1's first chunk is submitted to the GPU.
    # This overlaps CPU extraction work with GPU compute.
    # (layer_idx, cache_entry, tokens_to_extract)
    pending_extract: tuple[int, Any, int] | None = None

    def _flush_pending_extract() -> None:
        """Extract and enqueue KV for the previously-computed layer."""
        nonlocal pending_extract
        if pending_extract is None:
            return
        p_layer_idx: int
        p_cache: Any
        p_tokens: int
        p_layer_idx, p_cache, p_tokens = pending_extract  # pyright: ignore[reportAny]
        pending_extract = None

        t_extract_start = time.monotonic()
        k_np, v_np = extract_single_layer_kv(p_cache, p_tokens)  # pyright: ignore[reportAny]
        t_extract_end = time.monotonic()
        actual_tokens: int = int(k_np.shape[2])  # pyright: ignore[reportAny]  # (1, n_kv_heads, tokens, head_dim)
        logger.info(
            f"DISAGG_TIMING per_layer_extract_ms={(t_extract_end - t_extract_start) * 1000:.1f} "
            f"layer_idx={p_layer_idx} tokens={actual_tokens}"
        )
        try:
            send_queue.put((p_layer_idx, actual_tokens, k_np, v_np), timeout=120.0)
        except queue.Full:
            logger.warning(f"Per-layer KV enqueue timed out for layer {p_layer_idx}")
            error_event.set()

    for layer_idx, (layer, c, layer_type) in enumerate(
        zip(inner_model.layers, cache, layer_types, strict=True)
    ):
        # Process tokens in chunks through this single layer, mirroring
        # mlx_lm's _prefill pattern but one layer at a time.
        chunk_outputs: list[mx.array] = []
        remaining = x

        first_chunk_of_layer = True
        while remaining.shape[1] > 0:
            n = min(step, remaining.shape[1])
            x_chunk: mx.array = remaining[:, :n, :]
            remaining = remaining[:, n:, :]

            # Create mask for this chunk using current cache state —
            # cache.offset grows as each chunk is appended, so the mask
            # correctly allows attending to previously-processed chunks.
            if layer_type == "full_attention":
                mask = create_attention_mask(x_chunk, c)
            else:
                mask = create_attention_mask(
                    x_chunk,
                    c,
                    window_size=inner_model.window_size,
                )

            x_chunk = layer(x_chunk, mask, c)
            # Materialize both the layer output and KV cache for this chunk
            mx.eval(x_chunk, c.state)  # pyright: ignore[reportArgumentType]
            chunk_outputs.append(x_chunk)

            # After the first chunk of this layer is evaluated, the GPU is
            # idle momentarily — overlap by extracting the previous layer's
            # KV (pure CPU work) before the next mx.eval.
            if first_chunk_of_layer and pending_extract is not None:
                _flush_pending_extract()
                first_chunk_of_layer = False

        # Reassemble full sequence for next layer
        if len(chunk_outputs) == 1:
            x = chunk_outputs[0]
        else:
            x = mx.concatenate(chunk_outputs, axis=1)
            mx.eval(x)
        del chunk_outputs
        mx.clear_cache()

        if error_event.is_set():
            break

        # Send KVPS header on first layer (now cache keys are populated)
        if not header_sent:
            uses_bfloat16 = c.keys is not None and c.keys.dtype == mx.bfloat16  # pyright: ignore[reportUnnecessaryComparison]
            dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
            cache_n_kv_heads = c.keys.shape[1]
            cache_head_dim = c.keys.shape[3]
            header = struct.pack(
                "!4sBIBIII",
                _MAGIC,
                _VERSION,
                num_layers,
                dtype_flag,
                kv_send_cap,
                cache_n_kv_heads,
                cache_head_dim,
            )
            _sendall(sock, header)
            sender.start()
            header_sent = True

        # Defer KV extraction — it will be flushed while the next layer's
        # first chunk is computing on the GPU, overlapping CPU↔GPU work.
        tokens_to_extract = min(c.offset, kv_send_cap)
        pending_extract = (layer_idx, c, tokens_to_extract)

    # Flush the last layer's deferred extraction (no next layer to overlap with)
    if not error_event.is_set():
        _flush_pending_extract()

    # Apply norm and compute logits (needed for consistency but we discard the result)
    if not error_event.is_set():
        x = inner_model.norm(x)
        _logits: mx.array = model.lm_head(x)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        mx.eval(_logits)  # pyright: ignore[reportUnknownArgumentType]
        del _logits

    set_pipeline_prefill(model, is_prefill=False)

    t_prefill_end = time.monotonic()
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000

    # Send last_tokens frame
    if not error_event.is_set():
        last_tokens_np: np.ndarray[Any, Any] = np.array(last_tokens, copy=False).astype(
            np.int32
        )
        try:
            send_queue.put((len(last_tokens_np), last_tokens_np), timeout=10.0)
        except queue.Full:
            logger.warning("Per-layer KV last_tokens enqueue timed out")
            error_event.set()

    # Send end sentinel and wait for sender thread
    try:
        send_queue.put(None, timeout=10.0)
    except queue.Full:
        logger.warning("Per-layer KV send_queue full — sender thread likely stuck")
        error_event.set()

    sender.join(timeout=60.0)
    if sender.is_alive():
        logger.warning("Per-layer KV sender thread did not exit, closing socket")
        sock.close()
        sender.join(timeout=5.0)
    else:
        sock.close()

    if error_event.is_set():
        raise ConnectionError("Per-layer KV sender thread encountered a socket error")

    prefill_tps = num_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    logger.info(
        f"DISAGG_TIMING per_layer_prefill_ms={prefill_ms:.1f} "
        f"prefill_tps={prefill_tps:.1f} num_tokens={num_tokens} "
        f"layers_sent={num_layers} step_size={step}"
    )
    return prefill_tps, num_tokens


def send_kv_cache_per_layer_tp_sync(
    host: str,
    port: int,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt_tokens: mx.array,
    last_tokens: mx.array,
    cache: KVCacheType,
    sampler: Callable[[mx.array], mx.array],
    group: mx.distributed.Group,
    is_kv_sender: bool,
) -> tuple[float, int]:
    """Run per-layer TP prefill, NCCL all_gather per layer, stream KV to decode.

    Extends send_kv_cache_per_layer_sync for tensor-parallel: both ranks process
    each layer's full sequence in chunks, then do NCCL all_gather (~0.5ms) to merge
    the sharded KV heads.  Only the sender rank connects to decode and streams frames.

    This replaces send_kv_cache_pipelined_tp_relay_sync which used synchronous TCP
    relay (~250-600ms per exchange) inside the prefill callback, causing GPU idle gaps.

    Returns (prefill_tps, num_tokens).
    """
    from mlx_lm.models.base import (
        create_attention_mask,  # pyright: ignore[reportUnknownVariableType]
    )
    from mlx_lm.models.gpt_oss import GptOssMoeModel

    from exo.worker.engines.mlx.auto_parallel import set_pipeline_prefill

    num_layers = len(cache)
    num_tokens = len(prompt_tokens)
    kv_send_cap = max(1, num_tokens + 1 - len(last_tokens))
    step = _PER_LAYER_PREFILL_STEP_SIZE
    # --- Sender rank: connect to decode node and set up sender thread ---
    sock: socket.socket | None = None
    send_queue: queue.Queue[_LayerQueueItem] | None = None
    error_event = threading.Event()
    sender: threading.Thread | None = None

    if is_kv_sender:
        sock = _connect_with_retries(host, port)
        send_queue = queue.Queue(maxsize=4)
        sender = threading.Thread(
            target=_layer_sender_thread_fn,
            args=(sock, send_queue, error_event),
            daemon=True,
        )

    set_pipeline_prefill(model, is_prefill=True)

    t_prefill_start = time.monotonic()

    # --- Per-layer prefill with chunking + NCCL all_gather ---
    inner_model = cast(GptOssMoeModel, model.model)

    # Embed tokens and materialize
    x = inner_model.embed_tokens(prompt_tokens[None])
    mx.eval(x)

    header_sent = False
    layer_types: list[str] = cast(list[str], inner_model.layer_types)

    # Deferred extraction state: after gathering layer N, we defer extraction
    # until layer N+1's first chunk is evaluating on the GPU.
    # (layer_idx, full_k_gathered, full_v_gathered, tokens_to_extract)
    pending_extract: tuple[int, mx.array, mx.array, int] | None = None

    def _flush_pending_extract() -> None:
        """Extract and enqueue gathered KV for the previously-computed layer."""
        nonlocal pending_extract
        if pending_extract is None or send_queue is None:
            return
        p_layer_idx: int
        p_full_k: mx.array
        p_full_v: mx.array
        p_tokens: int
        p_layer_idx, p_full_k, p_full_v, p_tokens = pending_extract
        pending_extract = None

        t_extract_start = time.monotonic()
        k_np: np.ndarray[Any, Any] = _mlx_to_numpy(p_full_k[:, :, :p_tokens, :]).copy()
        v_np: np.ndarray[Any, Any] = _mlx_to_numpy(p_full_v[:, :, :p_tokens, :]).copy()
        del p_full_k, p_full_v
        t_extract_end = time.monotonic()
        actual_tokens: int = k_np.shape[2]  # pyright: ignore[reportAny]
        logger.info(
            f"DISAGG_TIMING per_layer_tp_extract_ms={(t_extract_end - t_extract_start) * 1000:.1f} "
            f"layer_idx={p_layer_idx} tokens={actual_tokens}"
        )
        try:
            send_queue.put((p_layer_idx, actual_tokens, k_np, v_np), timeout=120.0)
        except queue.Full:
            logger.warning(f"Per-layer TP KV enqueue timed out for layer {p_layer_idx}")
            error_event.set()

    for layer_idx, (layer, c, layer_type) in enumerate(
        zip(inner_model.layers, cache, layer_types, strict=True)
    ):
        # Process tokens in chunks through this single layer
        chunk_outputs: list[mx.array] = []
        remaining = x

        first_chunk_of_layer = True
        while remaining.shape[1] > 0:
            n = min(step, remaining.shape[1])
            x_chunk: mx.array = remaining[:, :n, :]
            remaining = remaining[:, n:, :]

            if layer_type == "full_attention":
                mask = create_attention_mask(x_chunk, c)
            else:
                mask = create_attention_mask(
                    x_chunk,
                    c,
                    window_size=inner_model.window_size,
                )

            x_chunk = layer(x_chunk, mask, c)
            mx.eval(x_chunk, c.state)  # pyright: ignore[reportArgumentType]
            chunk_outputs.append(x_chunk)

            # Overlap: flush previous layer's extract during GPU idle
            if first_chunk_of_layer and pending_extract is not None:
                _flush_pending_extract()
                first_chunk_of_layer = False

        # Reassemble full sequence for next layer
        if len(chunk_outputs) == 1:
            x = chunk_outputs[0]
        else:
            x = mx.concatenate(chunk_outputs, axis=1)
            mx.eval(x)
        del chunk_outputs
        mx.clear_cache()

        if error_event.is_set():
            break

        # --- NCCL all_gather this layer's KV (both ranks participate) ---
        state = c.state
        t_gather_start = time.monotonic()
        # state[0] shape: [1, n_kv_heads/world, seq, hd]
        # all_gather concatenates on dim 0 → [world, n_kv_heads/world, seq, hd]
        full_k = mx.distributed.all_gather(state[0], group=group)  # pyright: ignore[reportArgumentType]
        full_v = mx.distributed.all_gather(state[1], group=group)  # pyright: ignore[reportArgumentType]
        mx.eval(full_k, full_v)
        t_gather_end = time.monotonic()
        # Reshape to [1, n_kv_heads, seq, hd]
        full_k = full_k.reshape(1, -1, full_k.shape[2], full_k.shape[3])
        full_v = full_v.reshape(1, -1, full_v.shape[2], full_v.shape[3])
        logger.info(
            f"DISAGG_TIMING per_layer_tp_gather_ms={(t_gather_end - t_gather_start) * 1000:.1f} "
            f"layer_idx={layer_idx} shape={full_k.shape}"
        )

        if is_kv_sender:
            assert sock is not None
            assert send_queue is not None
            assert sender is not None

            # Send KVPS header on first layer (now we know shapes)
            if not header_sent:
                uses_bfloat16 = c.keys is not None and c.keys.dtype == mx.bfloat16  # pyright: ignore[reportUnnecessaryComparison]
                dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
                full_n_kv_heads = int(full_k.shape[1])
                cache_head_dim = int(full_k.shape[3])
                header = struct.pack(
                    "!4sBIBIII",
                    _MAGIC,
                    _VERSION,
                    num_layers,
                    dtype_flag,
                    kv_send_cap,
                    full_n_kv_heads,
                    cache_head_dim,
                )
                _sendall(sock, header)
                sender.start()
                header_sent = True

            # Defer extraction for overlap with next layer's compute
            tokens_to_extract = min(c.offset, kv_send_cap)
            pending_extract = (layer_idx, full_k, full_v, tokens_to_extract)
        # Non-sender: discard gathered result (participation in all_gather was required)
        else:
            del full_k, full_v

    # Flush the last layer's deferred extraction
    if is_kv_sender and not error_event.is_set():
        _flush_pending_extract()

    # Apply norm and compute logits (needed for consistency)
    if not error_event.is_set():
        x = inner_model.norm(x)
        _logits: mx.array = model.lm_head(x)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        mx.eval(_logits)  # pyright: ignore[reportUnknownArgumentType]
        del _logits

    set_pipeline_prefill(model, is_prefill=False)

    t_prefill_end = time.monotonic()
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000

    # --- Sender: finalize by sending last_tokens + end sentinel ---
    if is_kv_sender:
        assert sock is not None
        assert send_queue is not None
        assert sender is not None

        if not error_event.is_set():
            last_tokens_np: np.ndarray[Any, Any] = np.array(
                last_tokens, copy=False
            ).astype(np.int32)
            try:
                send_queue.put((len(last_tokens_np), last_tokens_np), timeout=10.0)
            except queue.Full:
                logger.warning("Per-layer TP KV last_tokens enqueue timed out")
                error_event.set()

        # Send end sentinel and wait for sender thread
        try:
            send_queue.put(None, timeout=10.0)
        except queue.Full:
            logger.warning("Per-layer TP KV send_queue full — sender thread likely stuck")
            error_event.set()

        sender.join(timeout=60.0)
        if sender.is_alive():
            logger.warning("Per-layer TP KV sender thread did not exit, closing socket")
            sock.close()
            sender.join(timeout=5.0)
        else:
            sock.close()

        if error_event.is_set():
            raise ConnectionError(
                "Per-layer TP KV sender thread encountered a socket error"
            )

    prefill_tps = num_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    logger.info(
        f"DISAGG_TIMING per_layer_tp_prefill_ms={prefill_ms:.1f} "
        f"prefill_tps={prefill_tps:.1f} num_tokens={num_tokens} "
        f"layers_sent={num_layers} step_size={step} is_kv_sender={is_kv_sender}"
    )
    return prefill_tps, num_tokens


def gather_sharded_kv_cache(
    cache: KVCacheType,
    group: mx.distributed.Group,
) -> None:
    """All-gather KV cache heads across tensor-parallel ranks, in-place.

    Before gather: each rank has [1, n_kv_heads/world, seq_len, head_dim]
    After gather: each rank has  [1, n_kv_heads, seq_len, head_dim]

    Uses c.state (not c.keys) to get only the valid data (trimmed to offset),
    since the raw c.keys buffer may have extra allocated space with garbage.
    """
    # Batch all all_gather calls and evaluate once to minimize CUDA graph
    # captures.  Per-layer mx.eval() creates ~60 separate graph entries
    # whose shapes vary with sequence length, polluting the graph cache.
    all_keys: list[mx.array] = []
    all_vals: list[mx.array] = []
    for c in cache:
        # c.state returns trimmed arrays: [1, n_kv_heads_per_rank, offset, head_dim]
        state = c.state
        all_keys.append(mx.distributed.all_gather(state[0], group=group))  # pyright: ignore[reportArgumentType]
        all_vals.append(mx.distributed.all_gather(state[1], group=group))  # pyright: ignore[reportArgumentType]
    mx.eval(*all_keys, *all_vals)
    # all_gather concatenates along dim 0 → [world, n_kv_heads/world, seq, hd]
    # Reshape to [1, n_kv_heads, seq, hd]
    for c, full_k, full_v in zip(cache, all_keys, all_vals, strict=True):
        c.state = (
            full_k.reshape(1, -1, full_k.shape[2], full_k.shape[3]),
            full_v.reshape(1, -1, full_v.shape[2], full_v.shape[3]),
        )


def send_precomputed_kv_cache_sync(
    host: str,
    port: int,
    cache: KVCacheType,
    last_tokens: mx.array,
) -> None:
    """Send an already-populated KV cache to the decode node using KVPS protocol.

    Unlike send_kv_cache_pipelined_sync which runs prefill internally, this function
    takes a pre-computed cache and streams it. Used by TP prefill after gather.
    """
    num_layers = len(cache)
    # Send fewer entries so the decode node can re-prefill the tail tokens
    # with its own model (same logic as the pipelined path).
    # Ensure at least 1 token is sent so the receiver always gets a chunk
    # frame — without this, short prompts (< DISAGG_REPREFILL_TOKENS) produce
    # num_tokens=0 and the receiver raises "No chunk frames received".
    num_tokens = max(1, cache[0].offset - max(0, len(last_tokens) - 2))

    sock = _connect_with_retries(host, port)

    # Send header
    uses_bfloat16 = cache[0].keys is not None and cache[0].keys.dtype == mx.bfloat16  # pyright: ignore[reportUnnecessaryComparison]
    dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
    cache_n_kv_heads = cache[0].keys.shape[1]
    cache_head_dim = cache[0].keys.shape[3]
    header = struct.pack(
        "!4sBIBIII",
        _MAGIC,
        _VERSION,
        num_layers,
        dtype_flag,
        num_tokens,
        cache_n_kv_heads,
        cache_head_dim,
    )
    _sendall(sock, header)

    # ── Phase 1: Extract all chunks to numpy (fast, compute-only) ──────
    # Extraction is a fast GPU→CPU copy that doesn't depend on the network.
    # By extracting ALL chunks first we can free the GPU KV cache before
    # any network I/O starts.  The previous approach interleaved extraction
    # with queue puts (maxsize=2), so the GPU cache stayed resident for the
    # entire network send duration — on unified-memory devices (GB10) this
    # meant the MLX arrays AND their numpy copies coexisted for seconds,
    # causing severe memory pressure and prefill TPS degradation across
    # consecutive runs.
    chunk_size = 4096
    chunk_index = 0
    t_send_start = time.monotonic()

    # Capture KV size before we free the cache arrays.
    kv_size_mb = sum(c.keys.nbytes + c.values.nbytes for c in cache) / 1024 / 1024

    extracted_chunks: list[KVChunkMessage] = []
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        delta = extract_kv_delta(cache, start, end, chunk_index)
        extracted_chunks.append(delta)
        chunk_index += 1

    # ── Phase 2: Free GPU KV cache immediately ──────────────────────────
    # All KV data is now in numpy.  Free the GPU-resident cache so the
    # sender rank releases MLX memory before any network I/O.
    for c in cache:
        c.state = (mx.zeros((1, 1, 1, 1)), mx.zeros((1, 1, 1, 1)))
    mx.synchronize()

    t_extract_end = time.monotonic()
    extract_ms = (t_extract_end - t_send_start) * 1000
    logger.info(
        f"DISAGG_TIMING kv_extract_ms={extract_ms:.1f} "
        f"chunks={chunk_index} kv_size_mb={kv_size_mb:.2f}"
    )

    # ── Phase 3: Drain numpy chunks through sender thread ────────────
    send_queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=2)
    error_event = threading.Event()
    sender = threading.Thread(
        target=_sender_thread_fn,
        args=(sock, send_queue, error_event),
        daemon=True,
    )
    sender.start()

    for delta in extracted_chunks:
        if error_event.is_set():
            break
        try:
            send_queue.put(delta, timeout=120.0)
        except queue.Full:
            logger.warning("Precomputed KV chunk enqueue timed out — sender stuck")
            error_event.set()
            break
    del extracted_chunks

    # Send last_tokens frame
    last_tokens_np: np.ndarray[Any, Any] = np.array(last_tokens, copy=False).astype(
        np.int32
    )
    try:
        send_queue.put((len(last_tokens_np), last_tokens_np), timeout=10.0)
    except queue.Full:
        logger.warning("Precomputed KV last_tokens enqueue timed out")
        error_event.set()

    # Send end sentinel and wait for sender thread.
    # Use a timeout on put() in case the sender thread is blocked on a
    # socket write (e.g. the receiver already disconnected and the kernel
    # buffer is full).
    try:
        send_queue.put(None, timeout=10.0)
    except queue.Full:
        logger.warning("Precomputed KV send_queue full — sender thread likely stuck")
        error_event.set()

    sender.join(timeout=60.0)
    if sender.is_alive():
        logger.warning("Precomputed KV sender thread did not exit, closing socket")
        # Closing the socket unblocks the sender thread's _sendall call.
        sock.close()
        sender.join(timeout=5.0)
    else:
        sock.close()

    t_send_end = time.monotonic()
    send_ms = (t_send_end - t_send_start) * 1000

    if error_event.is_set():
        raise ConnectionError("Precomputed KV sender thread encountered a socket error")

    logger.info(
        f"DISAGG_TIMING precomputed_kv_send_ms={send_ms:.1f} "
        f"kv_size_mb={kv_size_mb:.2f} chunks={chunk_index}"
    )


def send_kv_cache_pipelined_sync(
    host: str,
    port: int,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt_tokens: mx.array,
    last_tokens: mx.array,
    cache: KVCacheType,
    sampler: Callable[[mx.array], mx.array],
    prefill_step_size: int = 4096,
) -> tuple[float, int]:
    """Run prefill while streaming KV deltas to the decode node.

    Overlaps network transfer of each chunk with GPU compute of the next chunk.

    Returns (prefill_tps, num_tokens).
    """
    from exo.worker.engines.mlx.auto_parallel import set_pipeline_prefill
    from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE

    num_layers = len(cache)
    num_tokens = len(prompt_tokens)

    # Number of KV entries to send.  The decode node will re-prefill the tail
    # tokens (last_tokens) using its own model, so we only transfer entries
    # that the decode side will NOT recompute.
    #
    # Math: prompt has T total tokens.  Spark prefills prompt_tokens (T-1).
    # After stream_generate + trim(2), cache has T-2 entries.  Cap extraction
    # to kv_send_cap so the decode side receives exactly T - len(last_tokens)
    # entries, and stream_generate(prompt=last_tokens) recomputes the rest.
    kv_send_cap = max(1, num_tokens + 1 - len(last_tokens))

    # Connect to decode node (start connecting while we prefill)
    sock = _connect_with_retries(host, port)

    # Defer header send until after first prefill chunk populates the cache,
    # since fresh KVCache has keys=None before the first forward pass.
    header_sent = False

    # Set up sender thread
    send_queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=2)
    error_event = threading.Event()
    sender = threading.Thread(
        target=_sender_thread_fn,
        args=(sock, send_queue, error_event),
        daemon=True,
    )
    # sender.start() is deferred to on_prefill_chunk after header is sent

    # Track prefill progress for delta extraction
    prev_offset = 0
    chunk_index = 0
    t_prefill_start = time.monotonic()

    def on_prefill_chunk(processed: int, total: int) -> None:
        """Called after each mx.eval during prefill — extract and enqueue delta."""
        nonlocal prev_offset, chunk_index, header_sent
        if error_event.is_set():
            return

        # Cap extraction to kv_send_cap so we only send entries the decode
        # node won't recompute during its re-prefill of last_tokens.
        current_offset = min(cache[0].offset, kv_send_cap)
        if current_offset <= prev_offset:
            return

        # Send header on first chunk (cache keys are now populated)
        if not header_sent:
            uses_bfloat16 = (
                cache[0].keys is not None and cache[0].keys.dtype == mx.bfloat16  # pyright: ignore[reportUnnecessaryComparison]
            )
            dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
            # keys shape: [1, n_kv_heads, offset, head_dim]
            cache_n_kv_heads = cache[0].keys.shape[1]
            cache_head_dim = cache[0].keys.shape[3]
            header = struct.pack(
                "!4sBIBIII",
                _MAGIC,
                _VERSION,
                num_layers,
                dtype_flag,
                kv_send_cap,
                cache_n_kv_heads,
                cache_head_dim,
            )
            _sendall(sock, header)
            sender.start()
            header_sent = True

        t_extract_start = time.monotonic()
        delta = extract_kv_delta(cache, prev_offset, current_offset, chunk_index)
        t_extract_end = time.monotonic()
        logger.info(
            f"DISAGG_TIMING pipelined_extract_ms={(t_extract_end - t_extract_start) * 1000:.1f} "
            f"chunk_index={chunk_index} tokens={delta.num_tokens}"
        )

        send_queue.put(delta)
        prev_offset = current_offset
        chunk_index += 1

    set_pipeline_prefill(model, is_prefill=True)

    # Run prefill with progress callback that extracts + enqueues deltas
    # Use max_tokens=1 — we only care about filling the cache
    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=1,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=prefill_step_size,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
        prompt_progress_callback=on_prefill_chunk,
    ):
        break

    set_pipeline_prefill(model, is_prefill=False)

    t_prefill_end = time.monotonic()
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000

    # stream_generate added 1 extra generated token — trim it (and the preceding
    # token for arrays cache rollback consistency, matching prefill() behavior)
    for c in cache:
        c.trim(2)

    # If the header was never sent (e.g. callback never fired), send it now
    if not header_sent:
        uses_bfloat16 = cache[0].keys is not None and cache[0].keys.dtype == mx.bfloat16  # pyright: ignore[reportUnnecessaryComparison]
        dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
        cache_n_kv_heads = cache[0].keys.shape[1]
        cache_head_dim = cache[0].keys.shape[3]
        header = struct.pack(
            "!4sBIBIII",
            _MAGIC,
            _VERSION,
            num_layers,
            dtype_flag,
            kv_send_cap,
            cache_n_kv_heads,
            cache_head_dim,
        )
        _sendall(sock, header)
        sender.start()
        header_sent = True

    # Extract final delta if any tokens remain unsent after the last callback
    final_offset = min(cache[0].offset, kv_send_cap)
    if final_offset > prev_offset:
        delta = extract_kv_delta(cache, prev_offset, final_offset, chunk_index)
        send_queue.put(delta)
        logger.info(f"DISAGG_TIMING pipelined_final_chunk tokens={delta.num_tokens}")

    # Send last_tokens frame
    last_tokens_np: np.ndarray[Any, Any] = np.array(last_tokens, copy=False).astype(
        np.int32
    )
    send_queue.put((len(last_tokens_np), last_tokens_np))

    # Send end sentinel and wait for sender thread.
    try:
        send_queue.put(None, timeout=10.0)
    except queue.Full:
        logger.warning("Pipelined KV send_queue full — sender thread likely stuck")
        error_event.set()

    sender.join(timeout=30.0)
    if sender.is_alive():
        logger.warning("Pipelined KV sender thread did not exit, closing socket")
        sock.close()
        sender.join(timeout=5.0)
    else:
        sock.close()

    if error_event.is_set():
        raise ConnectionError("Pipelined sender thread encountered a socket error")

    prefill_tps = num_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    logger.info(
        f"DISAGG_TIMING pipelined_prefill_ms={prefill_ms:.1f} "
        f"prefill_tps={prefill_tps:.1f} num_tokens={num_tokens} "
        f"chunks_sent={chunk_index + 1}"
    )
    return prefill_tps, num_tokens


def send_kv_cache_pipelined_tp_sync(
    host: str,
    port: int,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt_tokens: mx.array,
    last_tokens: mx.array,
    cache: KVCacheType,
    sampler: Callable[[mx.array], mx.array],
    group: mx.distributed.Group,
    is_kv_sender: bool,
    prefill_step_size: int = 4096,
) -> tuple[float, int]:
    """Run TP prefill while streaming gathered KV deltas to the decode node.

    Both ranks call this function.  Inside the prefill callback, each chunk's
    sharded KV delta is all_gathered across ranks (NCCL collective — both ranks
    must participate).  The sender rank then extracts the full-headed delta to
    numpy and queues it for the background sender thread; the non-sender rank
    discards the gathered result.

    Returns (prefill_tps, num_tokens).
    """
    from exo.worker.engines.mlx.auto_parallel import set_pipeline_prefill
    from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE

    num_layers = len(cache)
    num_tokens = len(prompt_tokens)
    kv_send_cap = max(1, num_tokens + 1 - len(last_tokens))

    # Only the sender rank connects to the decode node.
    sock: socket.socket | None = None
    if is_kv_sender:
        sock = _connect_with_retries(host, port)

    header_sent = False

    # Sender thread (only used by sender rank)
    send_queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=2)
    error_event = threading.Event()
    sender: threading.Thread | None = None
    if is_kv_sender:
        assert sock is not None
        sender = threading.Thread(
            target=_sender_thread_fn,
            args=(sock, send_queue, error_event),
            daemon=True,
        )

    prev_offset = 0
    chunk_index = 0
    t_prefill_start = time.monotonic()

    def on_prefill_chunk(processed: int, total: int) -> None:
        """Called after each mx.eval during prefill — gather, extract, enqueue."""
        nonlocal prev_offset, chunk_index, header_sent
        if error_event.is_set():
            return

        current_offset = min(cache[0].offset, kv_send_cap)
        if current_offset <= prev_offset:
            return

        # Slice sharded KV deltas from each layer and all_gather across ranks.
        # This is a collective op — both ranks must execute it together.
        gathered_keys: list[mx.array] = []
        gathered_vals: list[mx.array] = []
        for c in cache:
            k_delta = c.keys[:, :, prev_offset:current_offset, :]
            v_delta = c.values[:, :, prev_offset:current_offset, :]
            gathered_keys.append(mx.distributed.all_gather(k_delta, group=group))
            gathered_vals.append(mx.distributed.all_gather(v_delta, group=group))
        mx.eval(*gathered_keys, *gathered_vals)

        t_gather_end = time.monotonic()

        if not is_kv_sender:
            # Non-sender: discard gathered data, update offset, return
            del gathered_keys, gathered_vals
            prev_offset = current_offset
            chunk_index += 1
            return

        assert sock is not None and sender is not None

        # Send header on first chunk
        if not header_sent:
            # After all_gather: [world, n_kv_heads/world, seq, hd] → full heads
            full_k0 = gathered_keys[0].reshape(
                1, -1, gathered_keys[0].shape[2], gathered_keys[0].shape[3]
            )
            uses_bfloat16 = full_k0.dtype == mx.bfloat16
            dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
            cache_n_kv_heads = full_k0.shape[1]
            cache_head_dim = full_k0.shape[3]
            header = struct.pack(
                "!4sBIBIII",
                _MAGIC,
                _VERSION,
                num_layers,
                dtype_flag,
                kv_send_cap,
                cache_n_kv_heads,
                cache_head_dim,
            )
            _sendall(sock, header)
            sender.start()
            header_sent = True

        # Extract gathered delta to numpy (full heads after reshape)
        t_extract_start = time.monotonic()
        delta_tokens = current_offset - prev_offset
        layer_keys_np: list[np.ndarray[Any, Any]] = []
        layer_vals_np: list[np.ndarray[Any, Any]] = []
        for gk, gv in zip(gathered_keys, gathered_vals, strict=True):
            # [world, n_kv_heads/world, tokens, hd] → [1, n_kv_heads, tokens, hd]
            full_k = gk.reshape(1, -1, gk.shape[2], gk.shape[3])
            full_v = gv.reshape(1, -1, gv.shape[2], gv.shape[3])
            layer_keys_np.append(_mlx_to_numpy(full_k).copy())
            layer_vals_np.append(_mlx_to_numpy(full_v).copy())
        del gathered_keys, gathered_vals

        t_extract_end = time.monotonic()
        logger.info(
            f"DISAGG_TIMING pipelined_tp_extract_ms={(t_extract_end - t_extract_start) * 1000:.1f} "
            f"gather_ms={(t_gather_end - t_extract_start) * 1000:.1f} "
            f"chunk_index={chunk_index} tokens={delta_tokens}"
        )

        delta = KVChunkMessage(
            chunk_index=chunk_index,
            start_offset=prev_offset,
            num_tokens=delta_tokens,
            layer_keys=layer_keys_np,
            layer_values=layer_vals_np,
        )
        send_queue.put(delta)
        prev_offset = current_offset
        chunk_index += 1

    set_pipeline_prefill(model, is_prefill=True)

    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=1,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=prefill_step_size,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
        prompt_progress_callback=on_prefill_chunk,
    ):
        break

    set_pipeline_prefill(model, is_prefill=False)

    t_prefill_end = time.monotonic()
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000

    for c in cache:
        c.trim(2)

    if is_kv_sender:
        assert sock is not None and sender is not None

        # If the header was never sent (callback never fired), send it now
        if not header_sent:
            # Need to gather once to get full head count
            gathered_keys_final: list[mx.array] = []
            gathered_vals_final: list[mx.array] = []
            for c in cache:
                state = c.state
                gathered_keys_final.append(
                    mx.distributed.all_gather(state[0], group=group)  # pyright: ignore[reportArgumentType]
                )
                gathered_vals_final.append(
                    mx.distributed.all_gather(state[1], group=group)  # pyright: ignore[reportArgumentType]
                )
            mx.eval(*gathered_keys_final, *gathered_vals_final)

            full_k0 = gathered_keys_final[0].reshape(
                1, -1, gathered_keys_final[0].shape[2], gathered_keys_final[0].shape[3]
            )
            uses_bfloat16 = full_k0.dtype == mx.bfloat16
            dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
            cache_n_kv_heads = full_k0.shape[1]
            cache_head_dim = full_k0.shape[3]
            header = struct.pack(
                "!4sBIBIII",
                _MAGIC,
                _VERSION,
                num_layers,
                dtype_flag,
                kv_send_cap,
                cache_n_kv_heads,
                cache_head_dim,
            )
            _sendall(sock, header)
            sender.start()
            header_sent = True
            del gathered_keys_final, gathered_vals_final

        # Extract final delta if any tokens remain unsent
        final_offset = min(cache[0].offset, kv_send_cap)
        if final_offset > prev_offset:
            # Final chunk: need one more all_gather
            gathered_keys_tail: list[mx.array] = []
            gathered_vals_tail: list[mx.array] = []
            for c in cache:
                k_delta = c.keys[:, :, prev_offset:final_offset, :]
                v_delta = c.values[:, :, prev_offset:final_offset, :]
                gathered_keys_tail.append(
                    mx.distributed.all_gather(k_delta, group=group)
                )
                gathered_vals_tail.append(
                    mx.distributed.all_gather(v_delta, group=group)
                )
            mx.eval(*gathered_keys_tail, *gathered_vals_tail)

            layer_keys_np_final: list[np.ndarray[Any, Any]] = []
            layer_vals_np_final: list[np.ndarray[Any, Any]] = []
            for gk, gv in zip(gathered_keys_tail, gathered_vals_tail, strict=True):
                full_k = gk.reshape(1, -1, gk.shape[2], gk.shape[3])
                full_v = gv.reshape(1, -1, gv.shape[2], gv.shape[3])
                layer_keys_np_final.append(_mlx_to_numpy(full_k).copy())
                layer_vals_np_final.append(_mlx_to_numpy(full_v).copy())
            del gathered_keys_tail, gathered_vals_tail

            delta = KVChunkMessage(
                chunk_index=chunk_index,
                start_offset=prev_offset,
                num_tokens=final_offset - prev_offset,
                layer_keys=layer_keys_np_final,
                layer_values=layer_vals_np_final,
            )
            send_queue.put(delta)
            logger.info(
                f"DISAGG_TIMING pipelined_tp_final_chunk tokens={delta.num_tokens}"
            )

        # Send last_tokens frame
        last_tokens_np: np.ndarray[Any, Any] = np.array(last_tokens, copy=False).astype(
            np.int32
        )
        send_queue.put((len(last_tokens_np), last_tokens_np))

        # Send end sentinel and wait for sender thread
        try:
            send_queue.put(None, timeout=10.0)
        except queue.Full:
            logger.warning(
                "Pipelined TP KV send_queue full — sender thread likely stuck"
            )
            error_event.set()

        sender.join(timeout=30.0)
        if sender.is_alive():
            logger.warning("Pipelined TP KV sender thread did not exit, closing socket")
            sock.close()
            sender.join(timeout=5.0)
        else:
            sock.close()

        if error_event.is_set():
            raise ConnectionError(
                "Pipelined TP sender thread encountered a socket error"
            )

    prefill_tps = num_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    logger.info(
        f"DISAGG_TIMING pipelined_tp_prefill_ms={prefill_ms:.1f} "
        f"prefill_tps={prefill_tps:.1f} num_tokens={num_tokens} "
        f"chunks_sent={chunk_index} is_sender={is_kv_sender}"
    )
    return prefill_tps, num_tokens


# ---------------------------------------------------------------------------
# TCP relay pipelined TP — replaces NCCL all_gather with inter-Spark TCP
# ---------------------------------------------------------------------------


def _relay_send_chunk(
    sock: socket.socket,
    chunk: KVChunkMessage,
) -> None:
    """Send a KV chunk shard to the sender rank over the fast inter-Spark link.

    Wire format per chunk: total_bytes(4) + chunk_index(4) + num_tokens(4)
    + per-layer [key_flat_bytes, val_flat_bytes].
    total_bytes == 0 is the end sentinel.
    """
    parts: list[bytes] = []
    for layer_idx in range(len(chunk.layer_keys)):
        parts.append(chunk.layer_keys[layer_idx].tobytes())
        parts.append(chunk.layer_values[layer_idx].tobytes())
    payload = b"".join(parts)
    header = struct.pack("!III", len(payload) + 12, chunk.chunk_index, chunk.num_tokens)
    _sendall(sock, header + payload)


def _relay_recv_chunk(
    conn: socket.socket,
    num_layers: int,
    n_kv_heads_per_rank: int,
    head_dim: int,
    wire_np_dtype: "type[np.uint16] | type[np.float16]",
) -> KVChunkMessage | None:
    """Receive a KV chunk shard from the non-sender rank.

    Returns None on end sentinel (total_bytes == 0).
    """
    hdr = _recvall(conn, 12)
    total_bytes: int
    chunk_index: int
    num_tokens: int
    total_bytes, chunk_index, num_tokens = struct.unpack("!III", hdr)  # pyright: ignore[reportAny]
    if total_bytes == 0:
        return None

    payload_len = total_bytes - 12
    payload = _recvall(conn, payload_len)

    # Parse per-layer key/value flat arrays
    elem_size = 2  # uint16 or float16
    layer_keys: list["np.ndarray[Any, Any]"] = []
    layer_values: list["np.ndarray[Any, Any]"] = []
    offset = 0
    for _ in range(num_layers):
        kv_flat_size = n_kv_heads_per_rank * num_tokens * head_dim * elem_size
        k_np: np.ndarray[Any, Any] = np.frombuffer(
            payload[offset : offset + kv_flat_size], dtype=wire_np_dtype
        ).copy()
        offset += kv_flat_size
        v_np: np.ndarray[Any, Any] = np.frombuffer(
            payload[offset : offset + kv_flat_size], dtype=wire_np_dtype
        ).copy()
        offset += kv_flat_size
        layer_keys.append(k_np)
        layer_values.append(v_np)

    return KVChunkMessage(
        chunk_index=chunk_index,
        start_offset=0,  # not used by caller
        num_tokens=num_tokens,
        layer_keys=layer_keys,
        layer_values=layer_values,
    )


def _combine_shards(
    my_delta: KVChunkMessage,
    other_delta: KVChunkMessage,
    my_rank: int,
    n_kv_heads_per_rank: int,
    num_layers: int,
) -> KVChunkMessage:
    """Concatenate two rank shards into a full-headed KV chunk."""
    combined_keys: list["np.ndarray[Any, Any]"] = []
    combined_vals: list["np.ndarray[Any, Any]"] = []
    for layer_idx in range(num_layers):
        my_k = my_delta.layer_keys[layer_idx].reshape(n_kv_heads_per_rank, -1)
        my_v = my_delta.layer_values[layer_idx].reshape(n_kv_heads_per_rank, -1)
        other_k = other_delta.layer_keys[layer_idx].reshape(n_kv_heads_per_rank, -1)
        other_v = other_delta.layer_values[layer_idx].reshape(n_kv_heads_per_rank, -1)

        if my_rank == 0:
            full_k = np.concatenate([my_k, other_k], axis=0)
            full_v = np.concatenate([my_v, other_v], axis=0)
        else:
            full_k = np.concatenate([other_k, my_k], axis=0)
            full_v = np.concatenate([other_v, my_v], axis=0)

        combined_keys.append(full_k.reshape(-1))
        combined_vals.append(full_v.reshape(-1))

    return KVChunkMessage(
        chunk_index=my_delta.chunk_index,
        start_offset=my_delta.start_offset,
        num_tokens=my_delta.num_tokens,
        layer_keys=combined_keys,
        layer_values=combined_vals,
    )


def send_kv_cache_pipelined_tp_relay_sync(
    decode_host: str,
    decode_port: int,
    relay_host: str,
    relay_port: int,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt_tokens: mx.array,
    last_tokens: mx.array,
    cache: KVCacheType,
    sampler: Callable[[mx.array], mx.array],
    is_kv_sender: bool,
    my_rank: int,
    world_size: int,
    prefill_step_size: int = 4096,
) -> tuple[float, int]:
    """Run TP prefill with TCP relay between Sparks instead of NCCL.

    Uses callback-based pipelining: each rank extracts its KV delta in the
    prefill callback, then the sender receives the other rank's shard via
    the fast ConnectX relay link (~254-373ms) and enqueues the combined
    chunk for the decode sender thread.  The non-sender sends its shard
    in the callback.  This overlaps relay exchange + decode network I/O
    with GPU compute for true pipelining.

    Returns (prefill_tps, num_tokens).
    """
    from exo.worker.engines.mlx.auto_parallel import set_pipeline_prefill
    from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE

    num_layers = len(cache)
    num_tokens = len(prompt_tokens)
    # Ensure at least 1 token is sent so the receiver always gets a chunk
    # frame — without this, short prompts (< DISAGG_REPREFILL_TOKENS) produce
    # kv_send_cap=0 and the receiver raises "No chunk frames received".
    kv_send_cap = max(1, num_tokens + 1 - len(last_tokens))

    # --- Relay connection setup ---
    relay_conn: socket.socket | None = None
    relay_listen: socket.socket | None = None

    if is_kv_sender:
        # Sender: listen for non-sender's relay connection
        relay_listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        relay_listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        relay_listen.bind((relay_host, relay_port))
        relay_listen.listen(1)
        relay_listen.settimeout(60.0)
        logger.info(f"KV relay server listening on {relay_host}:{relay_port}")
    else:
        # Non-sender: connect to sender's relay server (with retries)
        relay_conn = _connect_with_retries(
            relay_host, relay_port, retries=30, delay=1.0
        )
        logger.info(f"KV relay connected to {relay_host}:{relay_port}")

    # Sender: accept the non-sender's connection (after it's been established)
    if is_kv_sender:
        assert relay_listen is not None
        try:
            relay_conn, relay_addr = relay_listen.accept()  # pyright: ignore[reportAny]
            relay_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            relay_conn.settimeout(120.0)
            logger.info(f"KV relay accepted connection from {relay_addr}")
        finally:
            relay_listen.close()

    assert relay_conn is not None

    # --- Decode connection + sender thread (sender rank only) ---
    decode_sock: socket.socket | None = None
    decode_queue: queue.Queue[_QueueItem] | None = None
    decode_sender: threading.Thread | None = None
    error_event = threading.Event()

    if is_kv_sender:
        decode_sock = _connect_with_retries(decode_host, decode_port)
        decode_queue = queue.Queue(maxsize=2)
        decode_sender = threading.Thread(
            target=_sender_thread_fn,
            args=(decode_sock, decode_queue, error_event),
            daemon=True,
        )
        # decode_sender.start() deferred until after header is sent

    # --- Callback-based prefill with inline relay exchange ---
    prev_offset = 0
    chunk_index = 0
    header_sent = False
    t_prefill_start = time.monotonic()

    # Cache shape info (learned on first chunk)
    n_kv_heads_per_rank: int = 0
    head_dim_val: int = 0
    wire_np_dtype_val: type[np.uint16] | type[np.float16] = np.uint16

    def on_prefill_chunk(processed: int, total: int) -> None:
        nonlocal prev_offset, chunk_index, header_sent
        nonlocal n_kv_heads_per_rank, head_dim_val, wire_np_dtype_val

        if error_event.is_set():
            return

        current_offset = min(cache[0].offset, kv_send_cap)
        if current_offset <= prev_offset:
            return

        # Learn shapes from cache on first chunk
        if n_kv_heads_per_rank == 0:
            n_kv_heads_per_rank = cache[0].keys.shape[1]
            head_dim_val = cache[0].keys.shape[3]
            uses_bf16 = cache[0].keys is not None and cache[0].keys.dtype == mx.bfloat16  # pyright: ignore[reportUnnecessaryComparison]
            wire_np_dtype_val = np.uint16 if uses_bf16 else np.float16

        # Extract this rank's KV delta
        t_extract_start = time.monotonic()
        delta = extract_kv_delta(cache, prev_offset, current_offset, chunk_index)
        t_extract_end = time.monotonic()

        logger.info(
            f"DISAGG_TIMING pipelined_tp_relay_extract_ms={(t_extract_end - t_extract_start) * 1000:.1f} "
            f"chunk_index={chunk_index} tokens={delta.num_tokens}"
        )

        if is_kv_sender:
            assert relay_conn is not None
            assert decode_queue is not None
            assert decode_sender is not None

            # Send KVPS header to decode on first chunk
            if not header_sent:
                assert decode_sock is not None
                full_n_kv_heads = n_kv_heads_per_rank * world_size
                uses_bfloat16 = wire_np_dtype_val == np.uint16
                dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
                header = struct.pack(
                    "!4sBIBIII",
                    _MAGIC,
                    _VERSION,
                    num_layers,
                    dtype_flag,
                    kv_send_cap,
                    full_n_kv_heads,
                    head_dim_val,
                )
                _sendall(decode_sock, header)
                decode_sender.start()
                header_sent = True

            # Receive other rank's shard via fast relay link
            t_relay_start = time.monotonic()
            other_delta = _relay_recv_chunk(
                relay_conn,
                num_layers,
                n_kv_heads_per_rank,
                head_dim_val,
                wire_np_dtype_val,
            )
            t_relay_end = time.monotonic()
            assert other_delta is not None, "Unexpected end sentinel during relay recv"

            # Combine shards and enqueue for decode sender thread
            combined_msg = _combine_shards(
                delta, other_delta, my_rank, n_kv_heads_per_rank, num_layers
            )
            decode_queue.put(combined_msg)

            logger.info(
                f"DISAGG_TIMING pipelined_tp_relay_recv_ms={(t_relay_end - t_relay_start) * 1000:.1f} "
                f"chunk_index={chunk_index} tokens={delta.num_tokens}"
            )
        else:
            # Non-sender: send shard to sender via relay
            assert relay_conn is not None
            _relay_send_chunk(relay_conn, delta)

        prev_offset = current_offset
        chunk_index += 1

    set_pipeline_prefill(model, is_prefill=True)

    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=1,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=prefill_step_size,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
        prompt_progress_callback=on_prefill_chunk,
    ):
        break

    set_pipeline_prefill(model, is_prefill=False)

    t_prefill_end = time.monotonic()
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000

    for c in cache:
        c.trim(2)

    # Extract final delta if tokens remain unsent after last callback
    final_offset = min(cache[0].offset, kv_send_cap)
    if final_offset > prev_offset:
        if n_kv_heads_per_rank == 0:
            n_kv_heads_per_rank = cache[0].keys.shape[1]
            head_dim_val = cache[0].keys.shape[3]
            uses_bf16 = cache[0].keys is not None and cache[0].keys.dtype == mx.bfloat16  # pyright: ignore[reportUnnecessaryComparison]
            wire_np_dtype_val = np.uint16 if uses_bf16 else np.float16

        delta = extract_kv_delta(cache, prev_offset, final_offset, chunk_index)

        if is_kv_sender:
            assert relay_conn is not None
            assert decode_queue is not None
            assert decode_sender is not None
            assert decode_sock is not None

            if not header_sent:
                full_n_kv_heads = n_kv_heads_per_rank * world_size
                uses_bfloat16 = wire_np_dtype_val == np.uint16
                dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
                header = struct.pack(
                    "!4sBIBIII",
                    _MAGIC,
                    _VERSION,
                    num_layers,
                    dtype_flag,
                    kv_send_cap,
                    full_n_kv_heads,
                    head_dim_val,
                )
                _sendall(decode_sock, header)
                decode_sender.start()
                header_sent = True

            other_delta = _relay_recv_chunk(
                relay_conn,
                num_layers,
                n_kv_heads_per_rank,
                head_dim_val,
                wire_np_dtype_val,
            )
            assert other_delta is not None, "Unexpected end sentinel during relay recv"
            combined_msg = _combine_shards(
                delta, other_delta, my_rank, n_kv_heads_per_rank, num_layers
            )
            decode_queue.put(combined_msg)
        else:
            assert relay_conn is not None
            _relay_send_chunk(relay_conn, delta)

        chunk_index += 1

    # --- Finalize ---
    if is_kv_sender:
        assert decode_queue is not None
        assert decode_sender is not None
        assert decode_sock is not None

        # Send last_tokens frame
        last_tokens_np: np.ndarray[Any, Any] = np.array(last_tokens, copy=False).astype(
            np.int32
        )

        if header_sent:
            # Normal path: sender thread is running, enqueue last_tokens + end sentinel
            decode_queue.put((len(last_tokens_np), last_tokens_np))

            # Send end sentinel and wait for sender thread
            try:
                decode_queue.put(None, timeout=10.0)
            except queue.Full:
                logger.warning(
                    "Pipelined TP relay send_queue full — sender thread likely stuck"
                )
                error_event.set()

            decode_sender.join(timeout=60.0)
            if decode_sender.is_alive():
                logger.warning(
                    "Pipelined TP relay sender thread did not exit, closing socket"
                )
                decode_sock.close()
                decode_sender.join(timeout=5.0)
            else:
                decode_sock.close()

            if error_event.is_set():
                raise ConnectionError(
                    "Pipelined TP relay sender thread encountered a socket error"
                )
        else:
            # Short-prompt path: callback never fired, thread never started.
            # Send header + last_tokens + end directly on the socket.
            logger.info(
                "TP relay short-prompt path: sending header + data directly (no sender thread)"
            )
            uses_bfloat16 = cache[0].keys.dtype == mx.bfloat16
            dtype_flag = _DTYPE_BFLOAT16 if uses_bfloat16 else _DTYPE_FLOAT16
            cache_n_kv_heads = cache[0].keys.shape[1] * world_size
            cache_head_dim = cache[0].keys.shape[3]
            header = struct.pack(
                "!4sBIBIII",
                _MAGIC,
                _VERSION,
                num_layers,
                dtype_flag,
                kv_send_cap,
                cache_n_kv_heads,
                cache_head_dim,
            )
            _sendall(decode_sock, header)

            # Send last_tokens frame directly (matches _sender_thread_fn wire format)
            tok_bytes = last_tokens_np.astype(np.int32).tobytes()
            _sendall(
                decode_sock,
                struct.pack("!BI", _FRAME_LAST_TOKENS, len(last_tokens_np)) + tok_bytes,
            )

            # Send end sentinel
            _sendall(decode_sock, struct.pack("!B", _FRAME_END))
            decode_sock.close()
    else:
        # Non-sender: send end sentinel
        assert relay_conn is not None
        _sendall(relay_conn, struct.pack("!III", 0, 0, 0))

    # Close relay connection
    relay_conn.close()

    prefill_tps = num_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0.0
    logger.info(
        f"DISAGG_TIMING pipelined_tp_relay_prefill_ms={prefill_ms:.1f} "
        f"prefill_tps={prefill_tps:.1f} num_tokens={num_tokens} "
        f"chunks_sent={chunk_index} is_sender={is_kv_sender}"
    )
    return prefill_tps, num_tokens


# ---------------------------------------------------------------------------
# Pipelined receiver
# ---------------------------------------------------------------------------


def _recvall(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket."""
    parts: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 262144))
        if not chunk:
            raise ConnectionError(f"Connection closed with {remaining} bytes remaining")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _receive_pipelined(conn: socket.socket) -> tuple[list[KVCache], mx.array]:
    """Receive a pipelined KVPS stream from a connected socket.

    The 4-byte magic has already been consumed by the auto-detect logic.
    """
    # Read rest of header: version(1) + num_layers(4) + dtype_flag(1) + total_tokens(4)
    # v2 adds: n_kv_heads(4) + head_dim(4)
    rest_header = _recvall(conn, 10)
    version: int
    num_layers: int
    dtype_flag: int
    total_tokens: int
    version, num_layers, dtype_flag, total_tokens = struct.unpack("!BIBI", rest_header)  # pyright: ignore[reportAny]

    if version not in (0x01, _VERSION):
        raise ValueError(f"Unsupported KVPS version: {version}")

    # v2: read explicit n_kv_heads and head_dim from header
    header_n_kv_heads: int | None = None
    header_head_dim: int | None = None
    if version >= 0x02:
        extra = _recvall(conn, 8)
        header_n_kv_heads, header_head_dim = struct.unpack("!II", extra)  # pyright: ignore[reportAny]

    target_dtype = mx.bfloat16 if dtype_flag == _DTYPE_BFLOAT16 else mx.float16
    # bf16 is sent as raw uint16 bits (lossless bitcast); f16 as native float16
    wire_np_dtype = np.uint16 if dtype_flag == _DTYPE_BFLOAT16 else np.float16
    logger.info(
        f"Pipelined receive: {num_layers} layers, {total_tokens} tokens, "
        f"dtype={'bf16' if dtype_flag == _DTYPE_BFLOAT16 else 'f16'}"
        + (
            f", n_kv_heads={header_n_kv_heads}, head_dim={header_head_dim}"
            if header_n_kv_heads
            else ""
        )
    )

    # Pre-allocated buffers (lazily initialized on first chunk/layer frame)
    full_keys: list["np.ndarray[Any, Any]"] | None = None
    full_values: list["np.ndarray[Any, Any]"] | None = None
    head_dim_x_heads: int = 0
    n_kv_heads_wire: int = 0
    head_dim_wire: int = 0
    total_received_tokens: int = 0
    last_tokens: mx.array | None = None

    while True:
        frame_type_bytes = _recvall(conn, 1)
        frame_type = frame_type_bytes[0]

        if frame_type == _FRAME_END:
            break

        if frame_type == _FRAME_ERROR:
            msg_len: int = struct.unpack("!I", _recvall(conn, 4))[0]  # pyright: ignore[reportAny]
            msg = _recvall(conn, msg_len).decode("utf-8", errors="replace")
            raise RuntimeError(f"Prefill node reported error: {msg}")

        if frame_type == _FRAME_LAST_TOKENS:
            n_tokens_data = _recvall(conn, 4)
            n_tokens: int = struct.unpack("!I", n_tokens_data)[0]  # pyright: ignore[reportAny]
            tok_data = _recvall(conn, n_tokens * 4)
            last_tokens = mx.array(np.frombuffer(tok_data, dtype=np.int32).copy())
            continue

        if frame_type == _FRAME_LAYER:
            # Per-layer frame: [4B layer_idx][4B num_tokens][K data][V data]
            layer_hdr = _recvall(conn, 8)
            layer_frame_idx: int
            layer_num_tokens: int
            layer_frame_idx, layer_num_tokens = struct.unpack("!II", layer_hdr)  # pyright: ignore[reportAny]

            t_layer_start = time.monotonic()

            # Compute expected data size from header dimensions
            assert header_n_kv_heads is not None and header_head_dim is not None, (
                "Per-layer frames require v2 header with n_kv_heads and head_dim"
            )
            elem_size = 2  # uint16 or float16
            kv_data_size = (
                header_n_kv_heads * layer_num_tokens * header_head_dim * elem_size
            )

            k_data = _recvall(conn, kv_data_size)
            v_data = _recvall(conn, kv_data_size)

            layer_k_np: np.ndarray[Any, Any] = np.frombuffer(
                k_data, dtype=wire_np_dtype
            ).copy()
            layer_v_np: np.ndarray[Any, Any] = np.frombuffer(
                v_data, dtype=wire_np_dtype
            ).copy()

            # Initialize per-layer buffers on first frame
            if full_keys is None:
                head_dim_x_heads = header_n_kv_heads * header_head_dim
                n_kv_heads_wire = header_n_kv_heads
                head_dim_wire = header_head_dim
                # Allocate empty lists — each layer may have a different
                # token count (full_attention = all tokens, sliding_attention
                # = max_size from RotatingKVCache).
                full_keys = [
                    np.zeros(0, dtype=wire_np_dtype) for _ in range(num_layers)
                ]
                full_values = [
                    np.zeros(0, dtype=wire_np_dtype) for _ in range(num_layers)
                ]

            assert full_keys is not None and full_values is not None
            # Each layer frame carries that layer's tokens — store directly.
            # Token count varies per layer (RotatingKVCache stores fewer).
            full_keys[layer_frame_idx] = layer_k_np
            full_values[layer_frame_idx] = layer_v_np

            # Track max tokens across layers for total_received_tokens
            if layer_num_tokens > total_received_tokens:
                total_received_tokens = layer_num_tokens

            t_layer_end = time.monotonic()
            layer_size_mb = (len(k_data) + len(v_data)) / 1024 / 1024
            logger.info(
                f"DISAGG_TIMING per_layer_recv_ms={(t_layer_end - t_layer_start) * 1000:.1f} "
                f"layer_idx={layer_frame_idx} tokens={layer_num_tokens} "
                f"layer_mb={layer_size_mb:.2f}"
            )
            continue

        if frame_type == _FRAME_CHUNK:
            # Read chunk header: chunk_index(4) + start_offset(4) + num_tokens(4)
            chunk_hdr = _recvall(conn, 12)
            chunk_index: int
            start_offset: int
            chunk_num_tokens: int
            chunk_index, start_offset, chunk_num_tokens = struct.unpack(  # pyright: ignore[reportAny]
                "!III", chunk_hdr
            )

            t_chunk_start = time.monotonic()

            # Read per-layer KV data
            chunk_layer_keys: list["np.ndarray[Any, Any]"] = []
            chunk_layer_values: list["np.ndarray[Any, Any]"] = []
            for _layer_idx in range(num_layers):
                k_len_data = _recvall(conn, 4)
                k_len: int = struct.unpack("!I", k_len_data)[0]  # pyright: ignore[reportAny]
                k_data = _recvall(conn, k_len)

                v_len_data = _recvall(conn, 4)
                v_len: int = struct.unpack("!I", v_len_data)[0]  # pyright: ignore[reportAny]
                v_data = _recvall(conn, v_len)

                k_np: np.ndarray[Any, Any] = np.frombuffer(
                    k_data, dtype=wire_np_dtype
                ).copy()
                v_np: np.ndarray[Any, Any] = np.frombuffer(
                    v_data, dtype=wire_np_dtype
                ).copy()
                chunk_layer_keys.append(k_np)
                chunk_layer_values.append(v_np)

            # On first chunk, learn shapes and pre-allocate full buffers
            if full_keys is None:
                elements_per_array = len(chunk_layer_keys[0])
                head_dim_x_heads = elements_per_array // chunk_num_tokens
                # Derive per-head dimensions for head-major chunk assembly
                if header_n_kv_heads is not None and header_head_dim is not None:
                    n_kv_heads_wire = header_n_kv_heads
                    head_dim_wire = header_head_dim
                else:
                    # v1 fallback
                    head_dim_wire = head_dim_x_heads
                    n_kv_heads_wire = 1
                    for candidate in [128, 96, 80, 64]:
                        if head_dim_x_heads % candidate == 0:
                            head_dim_wire = candidate
                            n_kv_heads_wire = head_dim_x_heads // candidate
                            break
                full_keys = [
                    np.zeros(head_dim_x_heads * total_tokens, dtype=wire_np_dtype)
                    for _ in range(num_layers)
                ]
                full_values = [
                    np.zeros(head_dim_x_heads * total_tokens, dtype=wire_np_dtype)
                    for _ in range(num_layers)
                ]

            # Copy chunk data into pre-allocated buffers using per-head
            # placement.  The numpy arrays are in C-order of shape
            # (1, n_kv_heads, N, head_dim) — i.e. head-major.  A naive
            # flat copy (token-major) corrupts data when chunks have
            # different seq_len because the head boundaries don't align.
            assert full_keys is not None
            assert full_values is not None
            for layer_idx in range(num_layers):
                k_chunk = chunk_layer_keys[layer_idx].reshape(n_kv_heads_wire, -1)
                v_chunk = chunk_layer_values[layer_idx].reshape(n_kv_heads_wire, -1)
                s = start_offset * head_dim_wire
                e = s + chunk_num_tokens * head_dim_wire
                fk = full_keys[layer_idx].reshape(n_kv_heads_wire, -1)
                fv = full_values[layer_idx].reshape(n_kv_heads_wire, -1)
                fk[:, s:e] = k_chunk
                fv[:, s:e] = v_chunk

            total_received_tokens += chunk_num_tokens
            t_chunk_end = time.monotonic()
            chunk_size_mb = (
                sum(
                    len(k) * 2 + len(v) * 2
                    for k, v in zip(chunk_layer_keys, chunk_layer_values, strict=True)
                )
                / 1024
                / 1024
            )
            logger.info(
                f"DISAGG_TIMING pipelined_chunk_recv_ms={(t_chunk_end - t_chunk_start) * 1000:.1f} "
                f"chunk_index={chunk_index} chunk_tokens={chunk_num_tokens} "
                f"chunk_mb={chunk_size_mb:.2f}"
            )
            continue

        raise ValueError(f"Unknown KVPS frame type: {frame_type:#x}")

    if last_tokens is None:
        raise ValueError("No last_tokens frame received in pipelined stream")
    if full_keys is None or full_values is None:
        raise ValueError("No chunk or layer frames received in pipelined stream")

    # Determine n_kv_heads and head_dim
    if header_n_kv_heads is not None and header_head_dim is not None:
        n_kv_heads = header_n_kv_heads
        head_dim = header_head_dim
    else:
        # v1 fallback: guess from head_dim_x_heads
        head_dim = head_dim_x_heads
        n_kv_heads = 1
        for candidate_head_dim in [128, 96, 80, 64]:
            if head_dim_x_heads % candidate_head_dim == 0:
                head_dim = candidate_head_dim
                n_kv_heads = head_dim_x_heads // head_dim
                break

    t_convert_start = time.monotonic()
    caches: list[KVCache] = []
    elements_per_token = n_kv_heads * head_dim
    for layer_idx in range(num_layers):
        cache_entry = KVCache()
        layer_k = full_keys[layer_idx]
        layer_v = full_values[layer_idx]
        layer_n_tokens = (
            len(layer_k) // elements_per_token if elements_per_token > 0 else 0
        )

        # Sliding-attention layers (RotatingKVCache) only have max_size entries
        # (e.g. 128) instead of total_received_tokens.  Zero-pad them to the
        # full length, placing the real data at the END (it represents the most
        # recent tokens).  The sliding-window attention mask ensures the model
        # only attends to these real positions.
        if layer_n_tokens < total_received_tokens:
            pad_count = total_received_tokens - layer_n_tokens
            k_padded: np.ndarray[Any, Any] = np.zeros(
                total_received_tokens * elements_per_token, dtype=wire_np_dtype
            )
            v_padded: np.ndarray[Any, Any] = np.zeros(
                total_received_tokens * elements_per_token, dtype=wire_np_dtype
            )
            # Place real KV data at the tail (head-major: repeat per head)
            k_reshaped = layer_k.reshape(n_kv_heads, -1)
            v_reshaped = layer_v.reshape(n_kv_heads, -1)
            k_pad_reshaped = k_padded.reshape(n_kv_heads, -1)
            v_pad_reshaped = v_padded.reshape(n_kv_heads, -1)
            k_pad_reshaped[:, pad_count * head_dim :] = k_reshaped
            v_pad_reshaped[:, pad_count * head_dim :] = v_reshaped
            k_full = k_padded
            v_full = v_padded
        else:
            k_full = layer_k[: total_received_tokens * elements_per_token]
            v_full = layer_v[: total_received_tokens * elements_per_token]

        k_shaped: np.ndarray[Any, Any] = k_full.reshape(
            1, n_kv_heads, total_received_tokens, head_dim
        )
        v_shaped: np.ndarray[Any, Any] = v_full.reshape(
            1, n_kv_heads, total_received_tokens, head_dim
        )

        k_mx = mx.array(k_shaped)
        v_mx = mx.array(v_shaped)
        if dtype_flag == _DTYPE_BFLOAT16:
            # Bitcast uint16 bits back to bfloat16 — exact lossless recovery
            k_mx = k_mx.view(mx.bfloat16)
            v_mx = v_mx.view(mx.bfloat16)
        elif target_dtype == mx.float16 and k_mx.dtype != mx.float16:
            k_mx = k_mx.astype(mx.float16)
            v_mx = v_mx.astype(mx.float16)

        cache_entry.state = (k_mx, v_mx)
        cache_entry.offset = total_received_tokens
        caches.append(cache_entry)

    t_convert_end = time.monotonic()
    total_kv_mb = (
        sum(len(full_keys[i]) * 2 + len(full_values[i]) * 2 for i in range(num_layers))
        / 1024
        / 1024
    )
    logger.info(
        f"Pipelined receive complete: {num_layers} layers, "
        f"{total_received_tokens} tokens, {n_kv_heads} kv_heads, head_dim={head_dim}"
    )
    logger.info(
        f"DISAGG_TIMING numpy_to_mlx_ms={(t_convert_end - t_convert_start) * 1000:.1f} "
        f"kv_mb={total_kv_mb:.2f}"
    )
    return caches, last_tokens


# ---------------------------------------------------------------------------
# Auto-detecting receiver (handles both bulk and pipelined protocols)
# ---------------------------------------------------------------------------


async def receive_kv_cache_auto(port: int) -> tuple[list[KVCache], mx.array]:
    """Start a TCP server and auto-detect protocol from the first 4 bytes.

    If the first 4 bytes are b"KVPS", use the pipelined protocol.
    Otherwise, treat them as the start of the 8-byte bulk length header.
    """
    result_queue: asyncio.Queue[tuple[list[KVCache], mx.array]] = asyncio.Queue()

    async def handle_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        magic_or_header = await reader.readexactly(4)

        if magic_or_header == _MAGIC:
            # Pipelined protocol — hand off to synchronous receiver on a thread
            # since it uses blocking socket reads.
            raw_sock: socket.socket = writer.transport.get_extra_info("socket")  # pyright: ignore[reportAny]
            assert raw_sock is not None

            loop = asyncio.get_event_loop()
            caches, last_tokens = await loop.run_in_executor(
                None, _receive_pipelined, raw_sock
            )
            await result_queue.put((caches, last_tokens))
        else:
            # Bulk protocol — first 4 bytes are the high half of an 8-byte length
            rest = await reader.readexactly(4)
            length_bytes = magic_or_header + rest
            length: int = struct.unpack("!Q", length_bytes)[0]  # pyright: ignore[reportAny]
            kv_size_mb = length / 1024 / 1024
            logger.info(f"Receiving KV cache (bulk): {kv_size_mb:.1f} MB")

            t_recv_start = time.monotonic()
            data = await reader.readexactly(length)
            t_recv_end = time.monotonic()
            recv_ms = (t_recv_end - t_recv_start) * 1000
            logger.info(
                f"DISAGG_TIMING kv_network_recv_ms={recv_ms:.1f} "
                f"kv_size_mb={kv_size_mb:.2f}"
            )

            t_deser_start = time.monotonic()
            caches, last_tokens = deserialize_kv_cache(data)
            t_deser_end = time.monotonic()
            logger.info(
                f"DISAGG_TIMING kv_deserialize_ms={(t_deser_end - t_deser_start) * 1000:.1f}"
            )

            await result_queue.put((caches, last_tokens))

        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_connection, "0.0.0.0", port)
    logger.info(f"KV cache transfer server listening on port {port} (auto-detect)")

    async with server:
        caches, last_tokens = await result_queue.get()

    return caches, last_tokens


class KVTransferServer:
    """Persistent TCP server for receiving KV caches across multiple requests.

    Binds the port once at construction and accepts connections on each
    ``receive()`` call.  This eliminates the race between prefill sender
    and decode receiver that occurred when a fresh server was created per
    request — the port is always listening.
    """

    def __init__(self, port: int) -> None:
        self.port = port
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", port))
        self._server.listen(4)
        logger.info(f"KV transfer server listening on port {port} (persistent)")

    def receive(self, timeout: float = 120.0) -> tuple[list[KVCache], mx.array]:
        """Block until a prefill node connects and sends a KV cache.

        Raises ``TimeoutError`` if no connection arrives within *timeout*
        seconds, preventing the decode runner from blocking forever when
        the prefill node crashes or fails to connect.
        """
        self._server.settimeout(timeout)
        try:
            conn, addr = self._server.accept()  # pyright: ignore[reportAny]
        except socket.timeout:
            raise TimeoutError(
                f"No KV cache transfer received within {timeout}s on port {self.port}"
            ) from None
        finally:
            self._server.settimeout(None)
        logger.info(f"KV transfer connection from {addr}")
        # Set a recv timeout on the accepted connection so _recvall doesn't
        # block forever if the sender crashes mid-transfer.
        conn.settimeout(timeout)
        try:
            return _receive_one_connection(conn)
        finally:
            conn.close()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._server.close()
        logger.info(f"KV transfer server on port {self.port} closed")


def _receive_one_connection(
    conn: socket.socket,
) -> tuple[list[KVCache], mx.array]:
    """Auto-detect protocol and receive a KV cache from an accepted connection."""
    magic_or_header = _recvall(conn, 4)

    if magic_or_header == _MAGIC:
        return _receive_pipelined(conn)

    # Bulk protocol — first 4 bytes are the high half of an 8-byte length
    rest = _recvall(conn, 4)
    length_bytes = magic_or_header + rest
    length: int = struct.unpack("!Q", length_bytes)[0]  # pyright: ignore[reportAny]
    kv_size_mb = length / 1024 / 1024
    logger.info(f"Receiving KV cache (bulk): {kv_size_mb:.1f} MB")

    t_recv_start = time.monotonic()
    data = _recvall(conn, length)
    t_recv_end = time.monotonic()
    recv_ms = (t_recv_end - t_recv_start) * 1000
    logger.info(
        f"DISAGG_TIMING kv_network_recv_ms={recv_ms:.1f} kv_size_mb={kv_size_mb:.2f}"
    )

    t_deser_start = time.monotonic()
    caches, last_tokens = deserialize_kv_cache(data)
    t_deser_end = time.monotonic()
    logger.info(
        f"DISAGG_TIMING kv_deserialize_ms={(t_deser_end - t_deser_start) * 1000:.1f}"
    )
    return caches, last_tokens


def receive_kv_cache_auto_sync(port: int) -> tuple[list[KVCache], mx.array]:
    """One-shot synchronous receiver (legacy convenience wrapper).

    Creates a throwaway server, accepts one connection, and closes.
    Prefer ``KVTransferServer`` for production use.
    """
    server = KVTransferServer(port)
    try:
        return server.receive()
    finally:
        server.close()
