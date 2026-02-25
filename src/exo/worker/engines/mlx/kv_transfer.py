"""KV cache serialization and TCP transfer for disaggregated inference.

Handles serializing KV cache arrays from the prefill node and sending them
over a TCP socket to the decode node, which deserializes and reconstructs
the cache for token generation.
"""

import asyncio
import struct
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from exo.shared.types.mlx import KVCacheType
from exo.worker.runner.bootstrap import logger

# Port for direct KV cache transfer between nodes.
KV_TRANSFER_PORT = 52416


def _mlx_to_numpy(arr: mx.array) -> "np.ndarray[Any, Any]":
    """Convert an MLX array to numpy, handling bfloat16 which numpy doesn't support."""
    if arr.dtype == mx.bfloat16:
        return np.array(arr.astype(mx.float16), copy=False)
    return np.array(arr, copy=False)


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
    uses_bfloat16 = bool(loaded["uses_bfloat16"][0]) if "uses_bfloat16" in loaded else False  # pyright: ignore[reportAny]

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
