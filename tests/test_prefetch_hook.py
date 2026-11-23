"""The loader's prefetch hook: DevicePrefetcher runs it on device batches, NNUE builds it from
the feature transformer, and the feature set (not a constant) gives the tiled backward's list order."""

from types import SimpleNamespace

import pytest
import torch

from data_loader.dataset import DevicePrefetcher
from model import NNUE
from model.modules import ComposedFeatureTransformer, get_feature_cls
from model.modules.feature_transformer import tiled_ft_prep
from model.quantize import QuantizationConfig, QuantizationManager


def test_list_buckets_from_feature_set():
    ft = ComposedFeatureTransformer(
        get_feature_cls("Full_Threats+PP_3Wide+K32H3^"),
        64,
        QuantizationManager(QuantizationConfig()),
    )
    # K32H3 is last: its rows start after Full_Threats + PP_3Wide, 768 rows per bucket.
    assert ft.list_buckets() == (64368, 768)


def test_list_buckets_without_bucketed_tail():
    ft = ComposedFeatureTransformer(
        get_feature_cls("Full_Threats"), 64, QuantizationManager(QuantizationConfig())
    )
    assert ft.list_buckets() is None


def test_nnue_prefetch_hook_unpacks_batch():
    seen = []
    fake = SimpleNamespace(
        model=SimpleNamespace(
            input=SimpleNamespace(
                prefetch_preparer=lambda: lambda w, b: seen.append((w, b))
            )
        )
    )
    hook = NNUE.prefetch_hook(fake)
    batch = tuple(torch.full((1,), float(i)) for i in range(7))
    hook(batch)
    assert len(seen) == 1 and seen[0][0] is batch[2] and seen[0][1] is batch[3]

    fake.model.input.prefetch_preparer = lambda: None
    assert NNUE.prefetch_hook(fake) is None


class _Batches:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield (torch.full((4,), float(i)), torch.zeros(2, 3, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_hook_runs_on_device_batches_and_survives_errors():
    calls = []

    def hook(batch):
        calls.append(all(t.is_cuda for t in batch))
        if len(calls) == 2:
            raise RuntimeError("boom")

    ds = DevicePrefetcher(_Batches(4), num_batches=4, device="cuda", prefetch_hook=hook)
    with pytest.warns(RuntimeWarning, match="prefetch hook failed"):
        out = list(ds)
    # every batch arrives; the hook saw device tensors and was dropped after its first error
    assert [b[0][0].item() for b in out] == [0.0, 1.0, 2.0, 3.0]
    assert calls == [True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_parked_preparations_follow_their_batches(monkeypatch):
    """Every queued batch finds its preparation, however many are queued, and a batch that is
    freed unclaimed takes its entry with it."""

    def fake_prepare(white_indices, black_indices, buckets=None):
        t = torch.zeros(1, dtype=torch.int32, device=white_indices.device)
        return tiled_ft_prep.TiledPrep(2 * white_indices.shape[0], 1, t, t, t)

    monkeypatch.setattr(tiled_ft_prep, "prepare", fake_prepare)
    monkeypatch.setattr(tiled_ft_prep, "_parked", {})

    def batch():
        return torch.zeros(4, 8, dtype=torch.int32, device="cuda")

    queued = [batch() for _ in range(40)]
    for white in queued:
        tiled_ft_prep.register_prep(white, white)
    # claimed oldest first, as the trainer does, and only once
    assert all(tiled_ft_prep.claim_prep(white) is not None for white in queued)
    assert tiled_ft_prep.claim_prep(queued[0]) is None

    dropped = batch()
    tiled_ft_prep.register_prep(dropped, dropped)
    del dropped
    kept = batch()
    tiled_ft_prep.register_prep(kept, kept)
    assert len(tiled_ft_prep._parked) == 1
    assert tiled_ft_prep.claim_prep(kept) is not None
