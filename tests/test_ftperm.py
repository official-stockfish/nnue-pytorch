import copy
import io
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_check_eval import init_random_weights
import ftperm
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.utils.serialize import NNUEReader, NNUEWriter


def _generate_synthetic_batch(batch_size: int = 16, device: str = "cpu"):
    us = torch.randint(0, 2, (batch_size, 1), device=device)
    them = 1 - us
    # HalfKAv2_hm has 704 real features per bucket; generate active indices within bounds
    white_indices = torch.randint(0, 704, (batch_size, 30), dtype=torch.int32, device=device)
    black_indices = torch.randint(0, 704, (batch_size, 30), dtype=torch.int32, device=device)
    # Pad with -1
    white_indices = torch.cat(
        [white_indices, torch.full((batch_size, 2), -1, dtype=torch.int32, device=device)],
        dim=1,
    )
    black_indices = torch.cat(
        [black_indices, torch.full((batch_size, 2), -1, dtype=torch.int32, device=device)],
        dim=1,
    )
    c_us = torch.randint(0, 810810, (batch_size,), device=device)
    c_them = torch.randint(0, 810810, (batch_size,), device=device)
    return us, them, white_indices, black_indices, c_us, c_them


def test_ftperm_forward_parity():
    torch.manual_seed(42)
    np.random.seed(42)

    config = ModelConfig()
    model_orig = NNUEModel(feature_name=DEFAULT_FEATURES, config=config)
    init_random_weights(model_orig)
    model_orig.eval()

    model_orig.input.coalesce()
    model_orig.layer_stacks.coalesce_layer_stacks_inplace(model_orig.router.codebook)

    model_perm = copy.deepcopy(model_orig)
    perm = np.random.permutation(model_perm.L1 // 2)
    ftperm.ft_permute_impl(model_perm, perm)
    model_perm.eval()

    batch = _generate_synthetic_batch(batch_size=16)

    # 1. Unquantized forward parity
    with torch.no_grad():
        out_orig = model_orig.forward(*batch, fake_quantize_acts=False, fake_quantize_weights=False)
        out_perm = model_perm.forward(*batch, fake_quantize_acts=False, fake_quantize_weights=False)

    max_diff_float = (out_orig - out_perm).abs().max().item()
    assert max_diff_float < 1e-4, f"Float forward mismatch after ftperm: max diff {max_diff_float}"

    # 2. Fake-quantized forward parity
    with torch.no_grad():
        out_orig_q = model_orig.forward(*batch, fake_quantize_acts=True, fake_quantize_weights=True)
        out_perm_q = model_perm.forward(*batch, fake_quantize_acts=True, fake_quantize_weights=True)

    max_diff_q = (out_orig_q - out_perm_q).abs().max().item()
    assert max_diff_q < 1e-4, f"Quantized forward mismatch after ftperm: max diff {max_diff_q}"


def test_ftperm_serialization_roundtrip():
    torch.manual_seed(123)
    np.random.seed(123)

    config = ModelConfig()
    model = NNUEModel(feature_name=DEFAULT_FEATURES, config=config)
    init_random_weights(model)
    model.eval()

    model.clip_weights(include_input=True)
    model.input.coalesce()
    model.layer_stacks.coalesce_layer_stacks_inplace(model.router.codebook)

    perm = np.random.permutation(model.L1 // 2)
    ftperm.ft_permute_impl(model, perm)

    writer = NNUEWriter(model, description="FTPerm Test Net", verbose=False)
    buf = bytes(writer.buf)

    reader = NNUEReader(io.BytesIO(buf), DEFAULT_FEATURES, config)
    deserialized_model = reader.model
    deserialized_model.eval()

    batch = _generate_synthetic_batch(batch_size=16)

    with torch.no_grad():
        out_perm = model.forward(*batch, fake_quantize_acts=True, fake_quantize_weights=True)
        out_deserialized = deserialized_model.forward(
            *batch, fake_quantize_acts=True, fake_quantize_weights=True
        )

    max_diff = (out_perm - out_deserialized).abs().max().item()
    assert max_diff < 1e-4, f"Deserialized model mismatch: max diff {max_diff}"
