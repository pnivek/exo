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
from typing import Any, Callable

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
            keys = keys.astype(mx.bfloat16)
            values = values.astype(mx.bfloat16)
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
    host: str, port: int, retries: int = 30, delay: float = 1.0
) -> socket.socket:
    """Connect a TCP socket with retries."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((host, port))
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


def _sendall(sock: socket.socket, data: bytes | memoryview) -> None:
    """Send all bytes, raise on failure."""
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
            frame_data = _serialize_chunk_frame(item)
            t_send_start = time.monotonic()
            _sendall(sock, frame_data)
            t_send_end = time.monotonic()
            chunk_mb = len(frame_data) / 1024 / 1024
            logger.info(
                f"DISAGG_TIMING pipelined_chunk_send_ms={(t_send_end - t_send_start) * 1000:.1f} "
                f"chunk_index={item.chunk_index} chunk_mb={chunk_mb:.2f}"
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
    num_tokens = current_offset - prev_offset
    layer_keys: list["np.ndarray[Any, Any]"] = []
    layer_values: list["np.ndarray[Any, Any]"] = []
    for c in cache:
        # keys/values shape: [1, n_kv_heads, offset, head_dim]
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
    for c in cache:
        # c.state returns trimmed arrays: [1, n_kv_heads_per_rank, offset, head_dim]
        state = c.state
        full_k = mx.distributed.all_gather(state[0], group=group)  # pyright: ignore[reportArgumentType]
        full_v = mx.distributed.all_gather(state[1], group=group)  # pyright: ignore[reportArgumentType]
        mx.eval(full_k, full_v)
        # all_gather concatenates along dim 0 → [world, n_kv_heads/world, seq, hd]
        # Reshape to [1, n_kv_heads, seq, hd]
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
    num_tokens = cache[0].offset - max(0, len(last_tokens) - 2)

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

    # Set up sender thread
    send_queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=2)
    error_event = threading.Event()
    sender = threading.Thread(
        target=_sender_thread_fn,
        args=(sock, send_queue, error_event),
        daemon=True,
    )
    sender.start()

    # Extract and send the full cache in chunks
    chunk_size = 4096
    chunk_index = 0
    t_send_start = time.monotonic()
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        delta = extract_kv_delta(cache, start, end, chunk_index)
        send_queue.put(delta)
        chunk_index += 1

    # Send last_tokens frame
    last_tokens_np: np.ndarray[Any, Any] = np.array(last_tokens, copy=False).astype(
        np.int32
    )
    send_queue.put((len(last_tokens_np), last_tokens_np))

    # Send end sentinel and wait for sender thread
    send_queue.put(None)
    sender.join(timeout=60.0)
    sock.close()

    t_send_end = time.monotonic()
    send_ms = (t_send_end - t_send_start) * 1000
    kv_size_mb = sum(c.keys.nbytes + c.values.nbytes for c in cache) / 1024 / 1024

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
    kv_send_cap = num_tokens + 1 - len(last_tokens)

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

    # Send end sentinel and wait for sender thread
    send_queue.put(None)
    sender.join(timeout=30.0)

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

    # Pre-allocated buffers (lazily initialized on first chunk)
    full_keys: list["np.ndarray[Any, Any]"] | None = None
    full_values: list["np.ndarray[Any, Any]"] | None = None
    head_dim_x_heads: int = 0
    total_received_tokens: int = 0
    last_tokens: mx.array | None = None

    while True:
        frame_type_bytes = _recvall(conn, 1)
        frame_type = frame_type_bytes[0]

        if frame_type == _FRAME_END:
            break

        if frame_type == _FRAME_LAST_TOKENS:
            n_tokens_data = _recvall(conn, 4)
            n_tokens: int = struct.unpack("!I", n_tokens_data)[0]  # pyright: ignore[reportAny]
            tok_data = _recvall(conn, n_tokens * 4)
            last_tokens = mx.array(np.frombuffer(tok_data, dtype=np.int32).copy())
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
                full_keys = [
                    np.zeros(head_dim_x_heads * total_tokens, dtype=wire_np_dtype)
                    for _ in range(num_layers)
                ]
                full_values = [
                    np.zeros(head_dim_x_heads * total_tokens, dtype=wire_np_dtype)
                    for _ in range(num_layers)
                ]

            # Copy chunk data into pre-allocated buffers
            assert full_keys is not None
            assert full_values is not None
            for layer_idx in range(num_layers):
                chunk_elements = len(chunk_layer_keys[layer_idx])
                flat_start = start_offset * head_dim_x_heads
                flat_end = flat_start + chunk_elements
                full_keys[layer_idx][flat_start:flat_end] = chunk_layer_keys[layer_idx]
                full_values[layer_idx][flat_start:flat_end] = chunk_layer_values[
                    layer_idx
                ]

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
        raise ValueError("No chunk frames received in pipelined stream")

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

    caches: list[KVCache] = []
    for layer_idx in range(num_layers):
        cache_entry = KVCache()
        k_full = full_keys[layer_idx][: total_received_tokens * head_dim_x_heads]
        v_full = full_values[layer_idx][: total_received_tokens * head_dim_x_heads]

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

    logger.info(
        f"Pipelined receive complete: {num_layers} layers, "
        f"{total_received_tokens} tokens, {n_kv_heads} kv_heads, head_dim={head_dim}"
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
        self._server.listen(1)
        logger.info(f"KV transfer server listening on port {port} (persistent)")

    def receive(self) -> tuple[list[KVCache], mx.array]:
        """Block until a prefill node connects and sends a KV cache."""
        conn, addr = self._server.accept()  # pyright: ignore[reportAny]
        logger.info(f"KV transfer connection from {addr}")
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
