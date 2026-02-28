"""Tests for the KV transfer server and protocol handling.

Verifies that:
- KVTransferServer handles multiple sequential connections
- Bulk and pipelined protocols round-trip correctly
- Server port stays open between requests
"""

# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false
# ruff: noqa: E402
import socket
import struct
import threading
import time
from typing import Any

import numpy as np
import pytest

# These tests need MLX which may not be available in CI
pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.models.cache")

import mlx.core as mx
from mlx_lm.models.cache import KVCache  # noqa: I001

from exo.worker.engines.mlx.kv_transfer import (
    _DTYPE_FLOAT16,
    _FRAME_CHUNK,
    _FRAME_END,
    _FRAME_LAST_TOKENS,
    _MAGIC,
    _VERSION,
    KVTransferServer,
    deserialize_kv_cache,
    receive_kv_cache_auto_sync,
    serialize_kv_cache,
)


def _make_test_cache(
    num_layers: int = 2, seq_len: int = 4, n_kv_heads: int = 2, head_dim: int = 4
) -> tuple[list[KVCache], mx.array]:
    """Create a small test KV cache with known values."""
    caches: list[KVCache] = []
    for i in range(num_layers):
        c = KVCache()
        k = mx.ones((1, n_kv_heads, seq_len, head_dim), dtype=mx.float16) * (i + 1)
        v = mx.ones((1, n_kv_heads, seq_len, head_dim), dtype=mx.float16) * (i + 10)
        c.state = (k, v)
        caches.append(c)
    last_tokens = mx.array([42, 43])
    return caches, last_tokens


def _send_bulk(host: str, port: int, data: bytes) -> None:
    """Send a bulk KV cache payload to the server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    header = struct.pack("!Q", len(data))
    sock.sendall(header)
    sock.sendall(data)
    sock.close()


def _send_pipelined(
    host: str,
    port: int,
    caches: list[KVCache],
    last_tokens: mx.array,
) -> None:
    """Send a KV cache via the pipelined protocol."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    num_layers = len(caches)
    num_tokens = caches[0].offset
    n_kv_heads = caches[0].keys.shape[1]
    head_dim = caches[0].keys.shape[3]

    # Header
    header = struct.pack(
        "!4sBIBIII",
        _MAGIC,
        _VERSION,
        num_layers,
        _DTYPE_FLOAT16,
        num_tokens,
        n_kv_heads,
        head_dim,
    )
    sock.sendall(header)

    # One chunk with all tokens
    frame_header = struct.pack("!BIII", _FRAME_CHUNK, 0, 0, num_tokens)
    sock.sendall(frame_header)
    for c in caches:
        k_np: np.ndarray[Any, Any] = np.array(c.keys.astype(mx.float16), copy=False)
        v_np: np.ndarray[Any, Any] = np.array(c.values.astype(mx.float16), copy=False)
        k_flat = k_np.reshape(-1).tobytes()
        v_flat = v_np.reshape(-1).tobytes()
        sock.sendall(struct.pack("!I", len(k_flat)))
        sock.sendall(k_flat)
        sock.sendall(struct.pack("!I", len(v_flat)))
        sock.sendall(v_flat)

    # Last tokens
    tok_np: np.ndarray[Any, Any] = np.array(last_tokens, copy=False).astype(np.int32)
    sock.sendall(struct.pack("!BI", _FRAME_LAST_TOKENS, len(tok_np)))
    sock.sendall(tok_np.tobytes())

    # End
    sock.sendall(struct.pack("!B", _FRAME_END))
    sock.close()


def _find_free_port() -> int:
    """Find a free TCP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


class TestKVTransferServer:
    def test_persistent_server_multiple_bulk_requests(self) -> None:
        """Server handles multiple sequential bulk protocol connections."""
        port = _find_free_port()
        server = KVTransferServer(port)

        try:
            for i in range(3):
                caches, last_tokens = _make_test_cache(num_layers=2, seq_len=4 + i)
                data = serialize_kv_cache(caches, last_tokens)

                # Send in background thread
                t = threading.Thread(target=_send_bulk, args=("127.0.0.1", port, data))
                t.start()

                # Receive
                received_caches, received_tokens = server.receive()
                t.join(timeout=5.0)

                assert len(received_caches) == 2
                assert received_caches[0].offset == 4 + i
                assert np.array_equal(np.array(received_tokens), np.array(last_tokens))
        finally:
            server.close()

    def test_persistent_server_multiple_pipelined_requests(self) -> None:
        """Server handles multiple sequential pipelined protocol connections."""
        port = _find_free_port()
        server = KVTransferServer(port)

        try:
            for _i in range(3):
                caches, last_tokens = _make_test_cache(
                    num_layers=2, seq_len=4, n_kv_heads=2, head_dim=4
                )

                t = threading.Thread(
                    target=_send_pipelined,
                    args=("127.0.0.1", port, caches, last_tokens),
                )
                t.start()

                received_caches, _tokens = server.receive()
                t.join(timeout=5.0)

                assert len(received_caches) == 2
                assert received_caches[0].offset == 4
                # Verify values round-trip
                assert float(
                    received_caches[0].keys[0, 0, 0, 0].item()
                ) == pytest.approx(1.0)
                assert float(
                    received_caches[1].keys[0, 0, 0, 0].item()
                ) == pytest.approx(2.0)
        finally:
            server.close()

    def test_port_stays_open_between_requests(self) -> None:
        """Port is connectable between receive() calls (no rebind needed)."""
        port = _find_free_port()
        server = KVTransferServer(port)

        try:
            # First request
            caches, last_tokens = _make_test_cache()
            data = serialize_kv_cache(caches, last_tokens)
            t = threading.Thread(target=_send_bulk, args=("127.0.0.1", port, data))
            t.start()
            server.receive()
            t.join(timeout=5.0)

            # Small delay (simulates decode generation time)
            time.sleep(0.1)

            # Verify port is still connectable (this is the key test)
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.settimeout(1.0)
            # This should connect immediately since the server is still listening
            probe.connect(("127.0.0.1", port))
            probe.close()
            # Accept and discard the probe connection
            # (we need to drain it from the accept queue)
        finally:
            server.close()

    def test_legacy_one_shot_still_works(self) -> None:
        """receive_kv_cache_auto_sync() still works as a convenience wrapper."""
        port = _find_free_port()
        caches, last_tokens = _make_test_cache()
        data = serialize_kv_cache(caches, last_tokens)

        # Send after short delay (receiver needs to bind first)
        def delayed_send() -> None:
            time.sleep(0.2)
            _send_bulk("127.0.0.1", port, data)

        t = threading.Thread(target=delayed_send)
        t.start()

        received_caches, _tokens = receive_kv_cache_auto_sync(port)
        t.join(timeout=5.0)

        assert len(received_caches) == 2

    def test_bulk_roundtrip_values(self) -> None:
        """Bulk serialize/deserialize preserves KV cache values."""
        caches, last_tokens = _make_test_cache(
            num_layers=3, seq_len=8, n_kv_heads=4, head_dim=8
        )
        data = serialize_kv_cache(caches, last_tokens)
        restored_caches, restored_tokens = deserialize_kv_cache(data)

        assert len(restored_caches) == 3
        for i in range(3):
            assert restored_caches[i].offset == 8
            # Check key values: layer i should have value i+1
            k_val = float(restored_caches[i].keys[0, 0, 0, 0].item())
            assert k_val == pytest.approx(i + 1)
            # Check value values: layer i should have value i+10
            v_val = float(restored_caches[i].values[0, 0, 0, 0].item())
            assert v_val == pytest.approx(i + 10)

        assert np.array_equal(np.array(restored_tokens), np.array(last_tokens))

    def test_server_close_releases_port(self) -> None:
        """After close(), another server can bind the same port."""
        port = _find_free_port()
        server1 = KVTransferServer(port)
        server1.close()

        # Should not raise
        server2 = KVTransferServer(port)
        server2.close()
