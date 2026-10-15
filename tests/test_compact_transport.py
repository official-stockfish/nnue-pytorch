"""Check feature IDs, padding, and batch memory lifetime."""

import os
import sys
from typing import NoReturn

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import _native, _transport, stream


@pytest.mark.parametrize("batch_size", [1, 7, 16384])
def test_native_packing(batch_size: int) -> None:
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/8/4k3/8/3K4/8/8 b - - 0 1",
    ]
    batch = stream.get_sparse_batch_from_fens(
        "Full_Threats+PP_3Wide+HalfKAv2_hm",
        [fens[i % 2] for i in range(batch_size)],
        list(range(batch_size)),
        [1] * batch_size,
        [0] * batch_size,
    )
    try:
        native = batch.contents
        expected = native.get_tensors("cpu")
        nw, nb = native.num_active_white_features, native.num_active_black_features
        size = 4 * batch_size + 2 + nw + nb
        packed = torch.full((size + 2,), -123, dtype=torch.int32)
        pack = _native.c_lib.pack_sparse_batch
        assert not pack(batch, packed.data_ptr(), size - 1)
        assert torch.all(packed == -123)
        assert pack(batch, packed.data_ptr(), size)
        assert torch.all(packed[size:] == -123)
        wo = packed[: batch_size + 1]
        bo = packed[batch_size + 1 : 2 * (batch_size + 1)]
        wi = packed[2 * (batch_size + 1) : 2 * (batch_size + 1) + nw]
        bi = packed[2 * (batch_size + 1) + nw : 2 * (batch_size + 1) + nw + nb]
        assert wo[0] == bo[0] == 0
        assert wo[-1] == nw and bo[-1] == nb
        for offsets, ids, dense in ((wo, wi, expected[2]), (bo, bi, expected[3])):
            torch.testing.assert_close(offsets.diff().long(), (dense >= 0).sum(1))
            torch.testing.assert_close(ids, dense[dense >= 0])
        torch.testing.assert_close(
            packed[size - 2 * batch_size : size - batch_size].long(), expected[-2]
        )
        torch.testing.assert_close(
            packed[size - batch_size : size].long(), expected[-1]
        )
    finally:
        stream.destroy_sparse_batch(batch)


@pytest.mark.skipif(
    not torch.cuda.is_available() or _transport.unpack_sparse_indices is None,
    reason="CUDA and Triton required",
)
@pytest.mark.parametrize("active", [0, 1, 33, 288])
@pytest.mark.parametrize("separate_stream", [False, True])
def test_unpack_exact_padding(active: int, separate_stream: bool) -> None:
    batch = 17
    white = torch.full((batch, active), -1, dtype=torch.int32)
    black = torch.full_like(white, -1)
    for row in range(batch):
        for perspective, dense in enumerate((white, black)):
            count = (
                active if row == batch - 1 else (row * (perspective + 1)) % (active + 1)
            )
            dense[row, :count] = torch.arange(count, dtype=torch.int32) + row * 100000
    offsets = [
        torch.cat(
            (torch.zeros(1, dtype=torch.int32), (dense >= 0).sum(1).cumsum(0).int())
        )
        for dense in (white, black)
    ]
    ids = [dense[dense >= 0] for dense in (white, black)]
    pieces = torch.arange(batch, dtype=torch.int32)
    packed = torch.cat((*offsets, *ids, pieces, pieces + 10)).pin_memory()
    expected = torch.cat((white.flatten(), black.flatten(), pieces, pieces + 10))
    cuda_stream = (
        torch.cuda.Stream() if separate_stream else torch.cuda.current_stream()
    )
    with torch.cuda.stream(cuda_stream):
        if separate_stream:
            torch.cuda._sleep(2_000_000)
        uploaded = packed.to("cuda", non_blocking=True)
        assert _transport.unpack_sparse_indices is not None
        actual = _transport.unpack_sparse_indices(
            uploaded, batch, active, ids[0].numel(), ids[1].numel()
        )
        del uploaded, packed
        # Reuse transport storage while the first kernel is still pending.
        torch.empty_like(actual).fill_(42)
    cuda_stream.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or _transport.unpack_sparse_indices is None,
    reason="CUDA and Triton required",
)
@pytest.mark.parametrize("compact", [False, True])
def test_large_native_batch_survives_release(
    monkeypatch: pytest.MonkeyPatch, compact: bool
) -> None:
    size = 16384
    batch = stream.get_sparse_batch_from_fens(
        "Full_Threats+PP_3Wide+HalfKAv2_hm",
        ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * size,
        [1] * size,
        [1] * size,
        [0] * size,
    )
    calls: list[bool] = []
    unpack = _transport.unpack_sparse_indices
    assert unpack is not None

    def record(
        packed: torch.Tensor,
        batch_size: int,
        max_active: int,
        num_white: int,
        num_black: int,
    ) -> torch.Tensor:
        calls.append(True)
        return unpack(packed, batch_size, max_active, num_white, num_black)

    monkeypatch.setattr(_transport, "unpack_sparse_indices", record)
    try:
        expected = batch.contents.get_tensors("cpu")
        cuda_stream = torch.cuda.Stream()
        with torch.cuda.stream(cuda_stream):
            torch.cuda._sleep(2_000_000)
            actual = batch.contents.get_tensors(
                "cuda",
                use_pinned_memory=True,
                transport_policy=_transport.CompactTransportPolicy(enabled=compact),
            )
        assert calls == ([True] if compact else [])
    finally:
        stream.destroy_sparse_batch(batch)
    cuda_stream.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("pointer_backed", [False, True])
def test_python_batch_dense_transport(
    monkeypatch: pytest.MonkeyPatch, device: str, pointer_backed: bool
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def unexpected_pack(*args: object) -> NoReturn:
        pytest.fail("Python-created batches must not access private native fields")

    monkeypatch.setattr(_native.c_lib, "pack_sparse_batch", unexpected_pack)
    size, active = (16384 if device == "cuda" else 2), 288
    floats = np.arange(size * 3, dtype=np.float32)
    ints = np.arange(size * (2 * active + 2), dtype=np.int32)
    batch = _native.SparseBatch()
    batch.size, batch.max_active_features = size, active
    batch.is_white = floats.ctypes.data_as(
        _native.ctypes.POINTER(_native.ctypes.c_float)
    )
    batch.white = ints.ctypes.data_as(_native.ctypes.POINTER(_native.ctypes.c_int))
    if pointer_backed:
        batch = _native.ctypes.pointer(batch).contents
    got = batch.get_tensors(device, use_pinned_memory=device == "cuda")
    torch.testing.assert_close(
        got[2].cpu().flatten(), torch.from_numpy(ints[: size * active])
    )
