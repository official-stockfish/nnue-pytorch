"""The loader's prefetch hook: FixedNumBatchesDataset runs it on device batches, NNUE builds it from
the feature transformer, and the feature set (not a constant) gives the tiled backward's list order."""

from types import SimpleNamespace

import pytest
import torch

from data_loader.dataset import FixedNumBatchesDataset
from model import NNUE
from model.modules import ComposedFeatureTransformer, get_feature_cls
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

    ds = FixedNumBatchesDataset(
        _Batches(4), num_batches=4, device="cuda", prefetch_hook=hook
    )
    out = [ds[i] for i in range(4)]
    # every batch arrives; the hook saw device tensors and was dropped after its first error
    assert [b[0][0].item() for b in out] == [0.0, 1.0, 2.0, 3.0]
    assert calls == [True, True]
