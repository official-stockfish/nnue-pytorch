"""Packed feature IDs, their unpacking on the host and GPU, and batch memory lifetime."""

import os

import numpy as np
import pytest
import torch

from data_loader import _transport, stream
from data_loader.config import DataloaderSkipConfig

_SMALL_BINPACK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".pgo", "small.binpack"
)
_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "8/8/8/4k3/8/3K4/8/8 b - - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 9",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
]
_CUDA_TRANSPORT = pytest.mark.skipif(
    not torch.cuda.is_available() or _transport.unpack_sparse_indices is None,
    reason="CUDA and Triton required",
)


def _fen_batch(batch_size: int):
    return stream.get_sparse_batch_from_fens(
        "Full_Threats+PP_3Wide+K32H3",
        [_FENS[i % len(_FENS)] for i in range(batch_size)],
        [i - batch_size // 2 for i in range(batch_size)],
        [i % 7 for i in range(batch_size)],
        [i % 3 - 1 for i in range(batch_size)],
    )


def _packed(native) -> torch.Tensor:
    size = native.size
    n = (
        3 * size
        + 2
        + native.num_active_white_features
        + native.num_active_black_features
    )
    return torch.from_numpy(np.ctypeslib.as_array(native.packed, shape=(n,)).copy())


@pytest.mark.parametrize("batch_size", [1, 7, 16384])
def test_packed_layout_matches_host_rows(batch_size: int) -> None:
    batch = _fen_batch(batch_size)
    try:
        native = batch.contents
        _, _, white, black, _, _, stacks = native.get_tensors("cpu")
        nw, nb = native.num_active_white_features, native.num_active_black_features
        packed = _packed(native)
        wo = packed[: batch_size + 1]
        bo = packed[batch_size + 1 : 2 * (batch_size + 1)]
        wi = packed[2 * (batch_size + 1) : 2 * (batch_size + 1) + nw]
        bi = packed[2 * (batch_size + 1) + nw : 2 * (batch_size + 1) + nw + nb]
        assert wo[0] == bo[0] == 0
        assert wo[-1] == nw and bo[-1] == nb
        for offsets, ids, rows in ((wo, wi, white), (bo, bi, black)):
            assert rows.shape == (batch_size, native.max_active_features)
            # Active IDs first in every row, -1 padding after.
            assert torch.all((rows >= 0).int().diff(dim=1) <= 0)
            assert torch.all(rows[rows < 0] == -1)
            torch.testing.assert_close(offsets.diff().long(), (rows >= 0).sum(1))
            torch.testing.assert_close(ids, rows[rows >= 0])
        torch.testing.assert_close(packed[-batch_size:].long(), stacks)
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
    stacks = torch.arange(batch, dtype=torch.int32)
    packed = torch.cat((*offsets, *ids, stacks)).pin_memory()
    expected = torch.cat((white.flatten(), black.flatten(), stacks))
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


@_CUDA_TRANSPORT
@pytest.mark.parametrize("batch_size", [1000, 16384])
def test_device_batch_matches_host_batch(batch_size: int) -> None:
    """The packed transport and GPU unpack give exactly the host-unpacked tensors, with the batch
    freed while the copy is still queued behind other work."""
    batch = _fen_batch(batch_size)
    try:
        expected = batch.contents.get_tensors("cpu")
        cuda_stream = torch.cuda.Stream()
        with torch.cuda.stream(cuda_stream):
            torch.cuda._sleep(2_000_000)
            actual = batch.contents.get_tensors("cuda", use_pinned_memory=True)
    finally:
        stream.destroy_sparse_batch(batch)
    cuda_stream.synchronize()
    for got, want in zip(actual, expected):
        assert got.is_cuda
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unpinned_device_batch_unpacks_on_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object) -> torch.Tensor:
        pytest.fail("without pinned memory the GPU unpack must not be used")

    monkeypatch.setattr(_transport, "unpack_sparse_indices", unexpected)
    batch = _fen_batch(100)
    try:
        expected = batch.contents.get_tensors("cpu")
        actual = batch.contents.get_tensors("cuda", use_pinned_memory=False)
    finally:
        stream.destroy_sparse_batch(batch)
    for got, want in zip(actual, expected):
        assert got.is_cuda
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


@pytest.mark.skipif(
    not os.path.exists(_SMALL_BINPACK), reason="needs .pgo/small.binpack"
)
def test_stream_batches_are_padded_rows() -> None:
    s = stream.create_sparse_batch_stream(
        b"Full_Threats+PP_3Wide+K32H3",
        2,
        [_SMALL_BINPACK],
        1024,
        True,
        DataloaderSkipConfig(),
    )
    try:
        for _ in range(3):
            batch = stream.fetch_next_sparse_batch(s)
            try:
                _, _, white, black, *_ = batch.contents.get_tensors("cpu")
                assert (
                    white.shape
                    == black.shape
                    == (1024, batch.contents.max_active_features)
                )
                counts = (white >= 0).sum(1)
                assert int(counts.sum()) == batch.contents.num_active_white_features
                # Active IDs come first in each row, padding after.
                assert torch.all((white >= 0).int().diff(dim=1) <= 0)
            finally:
                stream.destroy_sparse_batch(batch)
    finally:
        stream.destroy_sparse_batch_stream(s)
