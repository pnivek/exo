"""Round-trip tests for the torch-side KV producer against the msgpack wire protocol.

In a disaggregated run the prefill node is vLLM/torch and the decode node is
MLX, so every KV byte crosses an engine boundary that neither engine's own
tests exercise. `engines/vllm/disaggregated/adapter.py` serializes tensors
(bitcasting bfloat16 through uint16, since numpy has no native bfloat16) and
`engines/mlx/disaggregated/adapter.py::bytes_to_array` reads them back. These
tests pin that contract, plus the framing and per-layer grouping rules the MLX
client (`engines/mlx/disaggregated/client.py`) depends on.

No sockets and no GPU: every test writes into a `BytesIO` and reads it back.

torch is only installed on the CUDA prefill nodes, so the torch classes skip
elsewhere. `TestNumpyProducer` mirrors the same bitcast with numpy standing in
for torch and runs everywhere, including CPU-only CI.
"""

from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType
from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt
import pytest

from exo.worker.disaggregated.protocol import (
    ArraysState,
    Done,
    DType,
    Header,
    KVChunk,
    TensorBlob,
    read_header,
    read_message,
    write_arrays_state,
    write_done,
    write_error,
    write_header,
    write_kv_chunk,
)

# =============================================================================
# Optional-dependency fixtures
# =============================================================================


@pytest.fixture(scope="module")
def torch_module() -> ModuleType:
    return pytest.importorskip("torch", reason="torch is only installed on CUDA nodes")


@pytest.fixture(scope="module")
def mlx_core() -> ModuleType:
    return pytest.importorskip("mlx.core")


@pytest.fixture(scope="module")
def mlx_adapter() -> ModuleType:
    _ = pytest.importorskip("mlx.core")
    return pytest.importorskip("exo.worker.engines.mlx.disaggregated.adapter")


@pytest.fixture(scope="module")
def vllm_adapter() -> ModuleType:
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("vllm")
    return pytest.importorskip("exo.worker.engines.vllm.disaggregated.adapter")


# =============================================================================
# Consumer-side decoding, mirroring engines/mlx/disaggregated/client.py
# =============================================================================


@dataclass(frozen=True)
class DecodedStream:
    header: Header
    kv_chunks: dict[int, list[KVChunk]]
    arrays: dict[int, list[TensorBlob]]
    total_tokens: int


def read_stream_like_client(payload: bytes) -> DecodedStream:
    """Decode a prefill stream the way `remote_prefill_fetch` does, minus the socket.

    Same loop as `engines/mlx/disaggregated/client.py`: KV chunks group by
    `layer_idx` in arrival order, `ArraysState` replaces a layer's blobs, an
    `ErrorMessage` raises `RuntimeError`, and a stream that ends without a
    `Done` frame raises `ConnectionError`.
    """
    stream = io.BytesIO(payload)
    header = read_header(stream)
    kv_by_layer: dict[int, list[KVChunk]] = defaultdict(list)
    arrays: dict[int, list[TensorBlob]] = {}
    total_tokens = 0
    done_seen = False

    while True:
        message = read_message(stream)
        if message is None:
            break
        if isinstance(message, KVChunk):
            kv_by_layer[message.layer_idx].append(message)
        elif isinstance(message, ArraysState):
            arrays[message.layer_idx] = message.arrays
        elif isinstance(message, Done):
            total_tokens = message.total_tokens
            done_seen = True
            break
        else:
            raise RuntimeError(
                f"Prefill server error [{message.code}]: {message.message}"
            )

    if not done_seen:
        raise ConnectionError("Prefill server closed before Done frame")

    return DecodedStream(
        header=header,
        kv_chunks=dict(kv_by_layer),
        arrays=arrays,
        total_tokens=total_tokens,
    )


# =============================================================================
# Producer-side mirrors of engines/vllm/disaggregated/adapter.py
#
# Duplicated rather than imported because that module imports vLLM, which is
# not installed on every machine that has torch. `TestVllmAdapterParity` pins
# the mirrors to the real implementations wherever vLLM is available.
# =============================================================================


def torch_dtype_to_wire(dtype: Any) -> DType:
    """Mirror of `adapter.torch_dtype_to_wire`."""
    import torch

    mapping: dict[Any, DType] = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype on wire: {dtype}")
    return mapping[dtype]


def tensor_to_wire_bytes(tensor: Any) -> bytes:
    """Mirror of `adapter.tensor_to_wire_bytes`."""
    import torch

    contiguous = tensor.detach().contiguous().cpu()
    if contiguous.dtype == torch.bfloat16:
        return bytes(contiguous.view(torch.uint16).numpy().tobytes())
    return bytes(contiguous.numpy().tobytes())


def to_bfloat16(tensor: Any) -> Any:
    """Mirror of `adapter.to_bf16`."""
    import torch

    if tensor.dtype == torch.uint8:
        tensor = tensor.view(torch.float8_e4m3fn)
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return tensor.to(torch.float32).to(torch.bfloat16)
    if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return tensor
    return tensor.to(torch.bfloat16)


def write_torch_kv_chunk(
    stream: BinaryIO, layer_idx: int, keys: Any, values: Any
) -> None:
    """Mirror of `adapter.write_kv_layer_chunk` for already-NHD tensors."""
    write_kv_chunk(
        stream,
        layer_idx=layer_idx,
        num_tokens=int(keys.shape[0]),
        n_heads=int(keys.shape[1]),
        head_dim=int(keys.shape[2]),
        dtype=torch_dtype_to_wire(keys.dtype),
        keys=tensor_to_wire_bytes(keys),
        values=tensor_to_wire_bytes(values),
    )


# =============================================================================
# Deterministic test data
# =============================================================================


def exact_bfloat16_float32(shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    """float32 values whose low 16 mantissa bits are zero — i.e. exactly bfloat16.

    Using exactly-representable values keeps every comparison in these tests
    an equality, never a tolerance.
    """
    size = int(np.prod(shape))
    raw = (np.arange(size, dtype=np.float32) * np.float32(0.37)) - np.float32(5.0)
    truncated = (raw.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    return truncated.reshape(shape)


def bfloat16_bits(values: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """The bfloat16 bit pattern of float32 values that are exactly bfloat16."""
    return (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def make_torch_nhd(
    torch_module: ModuleType,
    num_tokens: int,
    n_heads: int,
    head_dim: int,
    dtype: Any,
    offset: float = 0.0,
) -> Any:
    """A deterministic NHD-shaped tensor: [num_tokens, n_heads, head_dim]."""
    size = num_tokens * n_heads * head_dim
    values = (
        torch_module.arange(size, dtype=torch_module.float32) * 0.37 - 5.0 + offset
    ).reshape(num_tokens, n_heads, head_dim)
    return values.to(dtype)


def torch_to_float32_numpy(torch_module: ModuleType, tensor: Any) -> npt.NDArray[Any]:
    return tensor.to(torch_module.float32).numpy()


def mlx_to_float32_numpy(mlx_core: ModuleType, array: Any) -> npt.NDArray[Any]:
    return np.asarray(array.astype(mlx_core.float32))


# =============================================================================
# torch-free producer: the same bitcast convention, with numpy standing in
# =============================================================================


class TestNumpyProducer:
    """Runs everywhere. Covers the wire conventions the torch producer relies on."""

    def test_bfloat16_bitcast_preserves_bits(self) -> None:
        shape = (6, 4, 8)
        source = exact_bfloat16_float32(shape)
        source_bits = bfloat16_bits(source)

        stream = io.BytesIO()
        write_kv_chunk(
            stream,
            layer_idx=0,
            num_tokens=shape[0],
            n_heads=shape[1],
            head_dim=shape[2],
            dtype="bfloat16",
            keys=source_bits.tobytes(),
            values=source_bits[::-1].copy().tobytes(),
        )
        _ = stream.seek(0)

        chunk = read_message(stream)
        assert isinstance(chunk, KVChunk)
        assert chunk.shape == shape
        assert chunk.dtype == "bfloat16"

        decoded_bits = np.frombuffer(chunk.keys, dtype=np.uint16).reshape(shape)
        assert np.array_equal(decoded_bits, source_bits)

        # A uint16 bit pattern widened back into float32 recovers the value
        # exactly — this is what makes bfloat16 survive a numpy-less wire.
        widened = (decoded_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        assert np.array_equal(widened, source)

    @pytest.mark.parametrize(
        ("wire_dtype", "numpy_dtype"),
        [("float16", np.float16), ("float32", np.float32)],
    )
    def test_direct_dtype_roundtrip(
        self, wire_dtype: DType, numpy_dtype: type[np.floating[Any]]
    ) -> None:
        shape = (5, 2, 4)
        source = exact_bfloat16_float32(shape).astype(numpy_dtype)

        stream = io.BytesIO()
        write_kv_chunk(
            stream,
            layer_idx=2,
            num_tokens=shape[0],
            n_heads=shape[1],
            head_dim=shape[2],
            dtype=wire_dtype,
            keys=source.tobytes(),
            values=source.tobytes(),
        )
        _ = stream.seek(0)

        chunk = read_message(stream)
        assert isinstance(chunk, KVChunk)
        assert chunk.dtype == wire_dtype
        decoded = np.frombuffer(chunk.keys, dtype=numpy_dtype).reshape(shape)
        assert np.array_equal(decoded, source)

    def test_stream_groups_chunks_by_layer_and_skips_gaps(self) -> None:
        shape_per_chunk = [(4, 2, 3), (7, 2, 3), (2, 2, 3)]
        layer_zero = [exact_bfloat16_float32(s) for s in shape_per_chunk]

        stream = io.BytesIO()
        write_header(stream, Header(num_layers=4, dtype="float32"))
        for part in layer_zero:
            write_kv_chunk(
                stream,
                layer_idx=0,
                num_tokens=part.shape[0],
                n_heads=part.shape[1],
                head_dim=part.shape[2],
                dtype="float32",
                keys=part.tobytes(),
                values=part.tobytes(),
            )
        # Layer 1 is skipped entirely — a sliding-window layer can ship nothing.
        single = exact_bfloat16_float32((3, 2, 3))
        write_kv_chunk(
            stream,
            layer_idx=2,
            num_tokens=3,
            n_heads=2,
            head_dim=3,
            dtype="float32",
            keys=single.tobytes(),
            values=single.tobytes(),
        )
        write_done(stream, total_tokens=13)

        result = read_stream_like_client(stream.getvalue())
        assert sorted(result.kv_chunks) == [0, 2]
        assert len(result.kv_chunks[0]) == 3
        assert len(result.kv_chunks[2]) == 1
        assert result.total_tokens == 13

        rejoined = np.concatenate(
            [
                np.frombuffer(chunk.keys, dtype=np.float32).reshape(chunk.shape)
                for chunk in result.kv_chunks[0]
            ],
            axis=0,
        )
        assert np.array_equal(rejoined, np.concatenate(layer_zero, axis=0))

    def test_error_frame_after_header_raises(self) -> None:
        stream = io.BytesIO()
        write_header(stream, Header(num_layers=1, dtype="bfloat16"))
        write_error(stream, code=500, message="engine.step: RuntimeError()")

        with pytest.raises(RuntimeError, match=r"Prefill server error \[500\]"):
            _ = read_stream_like_client(stream.getvalue())

    def test_error_frame_before_header_loses_the_error(self) -> None:
        """Documents a producer/consumer mismatch, it is not the desired behavior.

        `VllmEngine.serve_prefill` writes its 503 refusals (engine busy, model
        runner not initialized) *before* the header. `Header` has no tag and
        all-default fields, and msgspec ignores unknown fields, so the client's
        `read_header` silently decodes the error frame as an empty `Header`.
        The code and message never reach the decode node; it sees only a
        generic "closed before Done frame".
        """
        stream = io.BytesIO()
        write_error(stream, code=503, message="engine busy")

        decoded_as_header = read_header(io.BytesIO(stream.getvalue()))
        assert decoded_as_header == Header()

        with pytest.raises(ConnectionError, match="closed before Done frame"):
            _ = read_stream_like_client(stream.getvalue())


# =============================================================================
# torch producer -> MLX consumer
# =============================================================================


@pytest.mark.usefixtures("torch_module")
class TestTorchTensorToWireBytes:
    def test_bfloat16_decodes_bit_identical(
        self, torch_module: ModuleType, mlx_core: ModuleType, mlx_adapter: ModuleType
    ) -> None:
        shape = (6, 4, 8)
        source = make_torch_nhd(torch_module, *shape, dtype=torch_module.bfloat16)

        decoded = mlx_adapter.bytes_to_array(
            tensor_to_wire_bytes(source), shape, "bfloat16"
        )
        assert decoded.dtype == mlx_core.bfloat16
        assert tuple(decoded.shape) == shape

        # bfloat16 -> float32 is lossless on both sides, so compare values...
        source_float32 = torch_to_float32_numpy(torch_module, source)
        assert np.array_equal(mlx_to_float32_numpy(mlx_core, decoded), source_float32)

        # ...and independently compare the raw bit patterns, so a dtype that
        # merely round-trips numerically (e.g. via float32) still fails here.
        expected_bits = (source_float32.view(np.uint32) >> np.uint32(16)).astype(
            np.uint16
        )
        decoded_bits = np.asarray(decoded.view(mlx_core.uint16))
        assert np.array_equal(decoded_bits, expected_bits)

    @pytest.mark.parametrize("wire_dtype", ["float16", "float32"])
    def test_direct_dtype_decodes_exactly(
        self,
        wire_dtype: DType,
        torch_module: ModuleType,
        mlx_core: ModuleType,
        mlx_adapter: ModuleType,
    ) -> None:
        shape = (5, 2, 4)
        torch_dtype = {
            "float16": torch_module.float16,
            "float32": torch_module.float32,
        }[wire_dtype]
        source = make_torch_nhd(torch_module, *shape, dtype=torch_dtype)

        decoded = mlx_adapter.bytes_to_array(
            tensor_to_wire_bytes(source), shape, wire_dtype
        )
        assert torch_dtype_to_wire(source.dtype) == wire_dtype
        assert np.array_equal(np.asarray(decoded), source.numpy())

    def test_permuted_tensor_is_serialized_in_nhd_order(
        self, torch_module: ModuleType, mlx_core: ModuleType, mlx_adapter: ModuleType
    ) -> None:
        """vLLM's HND cache layout reaches the wire via a non-contiguous permute."""
        n_heads, num_tokens, head_dim = 4, 6, 8
        hnd = make_torch_nhd(
            torch_module, n_heads, num_tokens, head_dim, dtype=torch_module.bfloat16
        )
        nhd = hnd.permute(1, 0, 2)
        assert not nhd.is_contiguous()

        shape = (num_tokens, n_heads, head_dim)
        decoded = mlx_adapter.bytes_to_array(
            tensor_to_wire_bytes(nhd), shape, "bfloat16"
        )
        assert np.array_equal(
            mlx_to_float32_numpy(mlx_core, decoded),
            torch_to_float32_numpy(torch_module, nhd),
        )


@pytest.mark.usefixtures("torch_module")
class TestTorchFramedStream:
    def test_multi_layer_multi_chunk_stream_reassembles(
        self, torch_module: ModuleType, mlx_core: ModuleType, mlx_adapter: ModuleType
    ) -> None:
        """One layer arrives as several chunks; layer 2 never arrives at all."""
        n_heads, head_dim = 4, 8
        chunked_layers = {0: [5, 7, 3], 3: [6, 4]}
        whole_layers = {1: [9]}
        keys_by_layer: dict[int, list[Any]] = {}
        values_by_layer: dict[int, list[Any]] = {}

        stream = io.BytesIO()
        write_header(
            stream,
            Header(
                request_id="prefill-1",
                model_id="Qwen/Qwen3-0.6B",
                num_layers=4,
                dtype="bfloat16",
                start_pos=0,
            ),
        )
        for layer_idx, token_counts in sorted({**chunked_layers, **whole_layers}.items()):
            keys_by_layer[layer_idx] = []
            values_by_layer[layer_idx] = []
            for chunk_index, num_tokens in enumerate(token_counts):
                keys = make_torch_nhd(
                    torch_module,
                    num_tokens,
                    n_heads,
                    head_dim,
                    dtype=torch_module.bfloat16,
                    offset=float(layer_idx * 10 + chunk_index),
                )
                values = keys * -1.0
                keys_by_layer[layer_idx].append(keys)
                values_by_layer[layer_idx].append(values)
                write_torch_kv_chunk(stream, layer_idx, keys, values)

        conv_state = make_torch_nhd(
            torch_module, 1, 2, 3, dtype=torch_module.float32
        ).reshape(2, 3)
        write_arrays_state(
            stream,
            layer_idx=5,
            arrays=[
                TensorBlob(
                    dtype="float32",
                    shape=(2, 3),
                    data=tensor_to_wire_bytes(conv_state),
                )
            ],
        )
        write_done(stream, total_tokens=15)

        result = read_stream_like_client(stream.getvalue())

        assert result.header.request_id == "prefill-1"
        assert result.header.num_layers == 4
        assert sorted(result.kv_chunks) == [0, 1, 3]
        assert 2 not in result.kv_chunks
        assert [len(result.kv_chunks[i]) for i in (0, 1, 3)] == [3, 1, 2]
        assert result.total_tokens == 15

        for layer_idx, chunks in result.kv_chunks.items():
            decoded = [mlx_adapter.chunk_to_mlx_nhd(chunk) for chunk in chunks]
            joined_keys = mlx_core.concatenate([k for k, _ in decoded], axis=0)
            joined_values = mlx_core.concatenate([v for _, v in decoded], axis=0)
            expected_keys = torch_module.cat(keys_by_layer[layer_idx], dim=0)
            expected_values = torch_module.cat(values_by_layer[layer_idx], dim=0)

            assert tuple(joined_keys.shape) == tuple(expected_keys.shape)
            assert np.array_equal(
                mlx_to_float32_numpy(mlx_core, joined_keys),
                torch_to_float32_numpy(torch_module, expected_keys),
            )
            assert np.array_equal(
                mlx_to_float32_numpy(mlx_core, joined_values),
                torch_to_float32_numpy(torch_module, expected_values),
            )

        assert sorted(result.arrays) == [5]
        blob = result.arrays[5][0]
        assert blob.shape == (2, 3)
        assert np.array_equal(
            np.asarray(mlx_adapter.blob_to_mlx(blob)), conv_state.numpy()
        )

    def test_arrays_state_replaces_per_layer_blobs(
        self, torch_module: ModuleType, mlx_core: ModuleType, mlx_adapter: ModuleType
    ) -> None:
        """Hybrid layers ship conv + ssm state instead of paged K/V."""
        conv_state = make_torch_nhd(
            torch_module, 1, 2, 4, dtype=torch_module.float32
        ).reshape(2, 4)
        ssm_state = make_torch_nhd(
            torch_module, 1, 1, 6, dtype=torch_module.bfloat16
        ).reshape(6)

        stream = io.BytesIO()
        write_header(stream, Header(num_layers=1, dtype="bfloat16"))
        write_arrays_state(
            stream,
            layer_idx=7,
            arrays=[
                TensorBlob(
                    dtype="float32", shape=(2, 4), data=tensor_to_wire_bytes(conv_state)
                ),
                TensorBlob(
                    dtype="bfloat16", shape=(6,), data=tensor_to_wire_bytes(ssm_state)
                ),
            ],
        )
        write_done(stream, total_tokens=0)

        result = read_stream_like_client(stream.getvalue())
        assert not result.kv_chunks
        conv_blob, ssm_blob = result.arrays[7]
        assert np.array_equal(
            np.asarray(mlx_adapter.blob_to_mlx(conv_blob)), conv_state.numpy()
        )
        assert np.array_equal(
            mlx_to_float32_numpy(mlx_core, mlx_adapter.blob_to_mlx(ssm_blob)),
            torch_to_float32_numpy(torch_module, ssm_state),
        )


@pytest.mark.usefixtures("torch_module")
class TestTorchFloat8:
    def test_uint8_viewed_float8_converts_to_bfloat16(
        self, torch_module: ModuleType, mlx_core: ModuleType, mlx_adapter: ModuleType
    ) -> None:
        """FP8 KV caches reach the connector as uint8; the wire is always bfloat16."""
        if not hasattr(torch_module, "float8_e4m3fn"):
            pytest.skip("torch build has no float8_e4m3fn")

        source = torch_module.tensor(
            [0.0, 0.5, -1.5, 3.0, -240.0, 12.0], dtype=torch_module.float32
        )
        as_float8 = source.to(torch_module.float8_e4m3fn)
        expected = as_float8.to(torch_module.float32).to(torch_module.bfloat16)

        converted = to_bfloat16(as_float8.view(torch_module.uint8))
        assert converted.dtype == torch_module.bfloat16
        assert bool(
            torch_module.isfinite(converted.to(torch_module.float32)).all()
        ), "float8 reinterpreted as uint8 produced non-finite values"
        assert bool(torch_module.equal(converted, expected))

        decoded = mlx_adapter.bytes_to_array(
            tensor_to_wire_bytes(converted), (int(converted.shape[0]),), "bfloat16"
        )
        assert np.array_equal(
            mlx_to_float32_numpy(mlx_core, decoded),
            torch_to_float32_numpy(torch_module, expected),
        )

    def test_float_dtypes_pass_through_unchanged(
        self, torch_module: ModuleType
    ) -> None:
        for dtype in (torch_module.bfloat16, torch_module.float16, torch_module.float32):
            source = make_torch_nhd(torch_module, 2, 2, 2, dtype=dtype)
            assert to_bfloat16(source).dtype == dtype


# =============================================================================
# The mirrors above must not drift from the real adapter
# =============================================================================


@pytest.mark.usefixtures("torch_module")
class TestVllmAdapterParity:
    @pytest.mark.parametrize("wire_dtype", ["bfloat16", "float16", "float32"])
    def test_tensor_to_wire_bytes_matches_mirror(
        self, wire_dtype: DType, torch_module: ModuleType, vllm_adapter: ModuleType
    ) -> None:
        torch_dtype = {
            "bfloat16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "float32": torch_module.float32,
        }[wire_dtype]
        source = make_torch_nhd(torch_module, 4, 3, 8, dtype=torch_dtype)
        assert vllm_adapter.tensor_to_wire_bytes(source) == tensor_to_wire_bytes(source)
        assert vllm_adapter.torch_dtype_to_wire(source.dtype) == wire_dtype

    def test_to_bf16_matches_mirror(
        self, torch_module: ModuleType, vllm_adapter: ModuleType
    ) -> None:
        if not hasattr(torch_module, "float8_e4m3fn"):
            pytest.skip("torch build has no float8_e4m3fn")
        as_uint8 = (
            torch_module.tensor([1.0, -2.5, 7.0], dtype=torch_module.float32)
            .to(torch_module.float8_e4m3fn)
            .view(torch_module.uint8)
        )
        assert bool(
            torch_module.equal(vllm_adapter.to_bf16(as_uint8), to_bfloat16(as_uint8))
        )

    def test_write_kv_layer_chunk_decodes_on_mlx_side(
        self, torch_module: ModuleType, mlx_core: ModuleType, mlx_adapter: ModuleType,
        vllm_adapter: ModuleType,
    ) -> None:
        """The real producer helper, straight into the real consumer helper."""
        keys = make_torch_nhd(torch_module, 5, 4, 8, dtype=torch_module.bfloat16)
        values = keys * -1.0

        stream = io.BytesIO()
        write_header(stream, Header(num_layers=1, dtype="bfloat16"))
        vllm_adapter.write_kv_layer_chunk(stream, 0, keys, values)
        write_done(stream, total_tokens=5)

        result = read_stream_like_client(stream.getvalue())
        decoded_keys, decoded_values = mlx_adapter.chunk_to_mlx_nhd(
            result.kv_chunks[0][0]
        )
        assert np.array_equal(
            mlx_to_float32_numpy(mlx_core, decoded_keys),
            torch_to_float32_numpy(torch_module, keys),
        )
        assert np.array_equal(
            mlx_to_float32_numpy(mlx_core, decoded_values),
            torch_to_float32_numpy(torch_module, values),
        )
