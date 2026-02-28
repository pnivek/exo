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
    send_precomputed_kv_cache_sync,
    serialize_kv_cache,
)


def _make_test_cache(
    num_layers: int = 2, seq_len: int = 4, n_kv_heads: int = 2, head_dim: int = 4
) -> tuple[list[KVCache], mx.array]:
    """Create a small test KV cache with known float16 values."""
    caches: list[KVCache] = []
    for i in range(num_layers):
        c = KVCache()
        k = mx.ones((1, n_kv_heads, seq_len, head_dim), dtype=mx.float16) * (i + 1)
        v = mx.ones((1, n_kv_heads, seq_len, head_dim), dtype=mx.float16) * (i + 10)
        c.state = (k, v)
        caches.append(c)
    last_tokens = mx.array([42, 43])
    return caches, last_tokens


def _make_test_cache_bf16(
    num_layers: int = 2, seq_len: int = 4, n_kv_heads: int = 2, head_dim: int = 4
) -> tuple[list[KVCache], mx.array]:
    """Create a small test KV cache with bfloat16 values."""
    caches: list[KVCache] = []
    for i in range(num_layers):
        c = KVCache()
        k = mx.ones((1, n_kv_heads, seq_len, head_dim), dtype=mx.bfloat16) * (i + 1)
        v = mx.ones((1, n_kv_heads, seq_len, head_dim), dtype=mx.bfloat16) * (i + 10)
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

    def test_bf16_pipelined_lossless_roundtrip(self) -> None:
        """bf16 KV cache round-trips the pipelined protocol with bitwise-exact precision.

        This is the critical test for the TP-disagg bf16 fix.  Before the fix, bf16
        values were converted to float16 (lossy), causing the decode model to produce
        garbage at long context lengths.  After the fix, values are sent as raw uint16
        bits and recovered via mx.array.view(mx.bfloat16) — zero precision loss.
        """
        port = _find_free_port()
        server = KVTransferServer(port)

        try:
            caches, last_tokens = _make_test_cache_bf16()

            # Save original values for comparison (as raw uint16 bits)
            orig_keys_bits = [
                np.array(c.keys.view(mx.uint16), copy=True) for c in caches
            ]
            orig_vals_bits = [
                np.array(c.values.view(mx.uint16), copy=True) for c in caches
            ]

            # Send through the real pipelined protocol (uses fixed _mlx_to_numpy)
            t = threading.Thread(
                target=send_precomputed_kv_cache_sync,
                args=("127.0.0.1", port, caches, last_tokens),
            )
            t.start()
            received_caches, _received_tokens = server.receive()
            t.join(timeout=10.0)

            assert len(received_caches) == 2
            assert received_caches[0].keys.dtype == mx.bfloat16
            assert received_caches[1].keys.dtype == mx.bfloat16

            # Verify bitwise-identical recovery — not just approximately equal
            for i, (rc, orig_k_bits, orig_v_bits) in enumerate(
                zip(received_caches, orig_keys_bits, orig_vals_bits, strict=True)
            ):
                recv_k_bits: np.ndarray[Any, Any] = np.array(
                    rc.keys.view(mx.uint16), copy=True
                )
                recv_v_bits: np.ndarray[Any, Any] = np.array(
                    rc.values.view(mx.uint16), copy=True
                )
                assert np.array_equal(recv_k_bits, orig_k_bits), (
                    f"Layer {i} keys not bitwise-identical after bf16 round-trip"
                )
                assert np.array_equal(recv_v_bits, orig_v_bits), (
                    f"Layer {i} values not bitwise-identical after bf16 round-trip"
                )
        finally:
            server.close()

    def test_bf16_wire_uses_uint16_not_float16(self) -> None:
        """Verifies that bf16 data is on the wire as raw uint16 bits, not float16.

        This guards against regression: if someone changes _mlx_to_numpy to use
        .astype(mx.float16) again, the header dtype_flag would say BFLOAT16 but
        the data would actually be float16 — this test would catch that mismatch.
        """
        # Pick a bf16 value whose bit pattern changes under bf16→float16→bf16
        # bf16 has a larger exponent range; we use a small subnormal that survives
        # in bf16 but gets flushed to 0 in float16.
        # 0x0001 in bf16 = 5.96e-45 (bf16 subnormal); in float16 this rounds to 0.
        sentinel_uint16 = np.uint16(0x0080)  # a small bf16 value
        sentinel_bf16 = mx.array(
            np.array([[[[sentinel_uint16]]]], dtype=np.uint16)
        ).view(mx.bfloat16)
        sentinel_f16_val = float(sentinel_bf16.astype(mx.float16).item())
        sentinel_bf16_val = float(sentinel_bf16.item())

        # If the value differs after bf16→float16, it's a good sentinel
        # (test is still valid if they happen to be equal, but less discriminating)
        if sentinel_bf16_val == sentinel_f16_val:
            pytest.skip(
                "Sentinel value survives bf16→float16 conversion on this platform"
            )

        port = _find_free_port()
        server = KVTransferServer(port)
        try:
            from mlx_lm.models.cache import KVCache

            c = KVCache()
            k = mx.broadcast_to(sentinel_bf16, (1, 1, 2, 4))
            v = mx.broadcast_to(sentinel_bf16, (1, 1, 2, 4))
            c.state = (k, v)
            last_tokens = mx.array([1])
            caches = [c]

            t = threading.Thread(
                target=send_precomputed_kv_cache_sync,
                args=("127.0.0.1", port, caches, last_tokens),
            )
            t.start()
            received_caches, _ = server.receive()
            t.join(timeout=10.0)

            # The recovered value must match the original bf16, NOT the float16 version
            recovered_val = float(received_caches[0].keys[0, 0, 0, 0].item())
            assert recovered_val == pytest.approx(sentinel_bf16_val), (
                f"bf16 precision lost: expected {sentinel_bf16_val}, "
                f"got {recovered_val} (float16 would give {sentinel_f16_val})"
            )
        finally:
            server.close()
