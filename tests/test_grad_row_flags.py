import gc

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from model.modules.feature_transformer import grad_row_flags as grf


def test_lookup_matches_aliases_until_modified():
    g = torch.zeros(6, 4)
    flags = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.uint8)
    grf.register(g, flags)
    # a new TensorImpl over the same memory, as under torch.compile
    alias = g.view(6, 4)
    assert grf.lookup(alias) is flags
    assert grf.lookup(torch.zeros(6, 4)) is None  # other memory
    assert grf.lookup(g[:3]) is None  # other shape
    alias.add_(1.0)  # an in-place change bumps the shared version counter
    assert grf.lookup(g) is None


def test_registration_does_not_keep_the_gradient_alive():
    g = torch.zeros(1024, 64)
    grf.register(g, torch.ones(1024, dtype=torch.uint8))
    storage = StorageWeakRef(g.untyped_storage())
    del g
    gc.collect()
    assert storage.expired()
    # a new tensor that happens to get the same address must not match the dead registration
    assert grf.lookup(torch.zeros(1024, 64)) is None
