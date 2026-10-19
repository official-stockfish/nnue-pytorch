import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ftperm import ft_permute_impl
from model import NNUE, ModelConfig, NNUEConfig


def test_ft_permute_preserves_evaluation_and_preserves_dyn_bias():
    torch.manual_seed(42)
    np.random.seed(42)

    config = NNUEConfig(
        features="HalfKAv2_hm^",
    )
    nnue = NNUE(config=config)
    model = nnue.model
    model.eval()

    batch_size = 4
    num_active = 30
    us = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    them = 1.0 - us
    white_indices = torch.randint(
        0, model.input.NUM_INPUTS, (batch_size, num_active), dtype=torch.int32
    )
    black_indices = torch.randint(
        0, model.input.NUM_INPUTS, (batch_size, num_active), dtype=torch.int32
    )
    piece_count = torch.tensor([[16], [20], [8], [32]], dtype=torch.int32)
    queen_bucket = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32)

    with torch.no_grad():
        out_before = model(
            us,
            them,
            white_indices,
            black_indices,
            piece_count,
            queen_bucket,
            fake_quantize_acts=False,
            fake_quantize_weights=False,
        )

    l1_size = model.L1
    dyn_weights_before = [f.weight[:, l1_size:].clone() for f in model.input.features]
    dyn_biases_before = model.input.bias[l1_size:].clone()

    perm = np.random.permutation(l1_size // 2)
    ft_permute_impl(model, perm)

    # Dynamic bias parameters MUST remain identical
    for f, orig_w in zip(model.input.features, dyn_weights_before, strict=True):
        assert torch.equal(f.weight[:, l1_size:], orig_w), (
            "Dynamic bias weights were modified by ft_permute_impl!"
        )
    assert torch.equal(model.input.bias[l1_size:], dyn_biases_before), (
        "Dynamic bias biases were modified by ft_permute_impl!"
    )

    # Evaluation should be identical
    with torch.no_grad():
        out_after = model(
            us,
            them,
            white_indices,
            black_indices,
            piece_count,
            queen_bucket,
            fake_quantize_acts=False,
            fake_quantize_weights=False,
        )

    torch.testing.assert_close(
        out_before,
        out_after,
        rtol=1e-5,
        atol=1e-5,
        msg="Model evaluation changed after feature permutation!",
    )


def test_custom_neuron_counts():
    """Verify that models with non-standard neuron counts (e.g. L1=512, L2=16, L3=16) work without hardcoded 1024."""
    config = NNUEConfig(
        features="HalfKAv2_hm^",
        model_config=ModelConfig(
            L1=512,
            L1_bias=256,
            L2=16,
            L3=16,
        ),
    )
    nnue = NNUE(config=config)
    model = nnue.model
    model.eval()

    assert model.input.num_outputs == 512 + 256
    assert model.input.l1_size == 512

    batch_size = 2
    num_active = 20
    us = torch.tensor([[1.0], [0.0]])
    them = 1.0 - us
    white_indices = torch.randint(
        0, model.input.NUM_INPUTS, (batch_size, num_active), dtype=torch.int32
    )
    black_indices = torch.randint(
        0, model.input.NUM_INPUTS, (batch_size, num_active), dtype=torch.int32
    )
    piece_count = torch.tensor([[16], [20]], dtype=torch.int32)
    queen_bucket = torch.tensor([[0], [1]], dtype=torch.int32)

    with torch.no_grad():
        out = model(
            us,
            them,
            white_indices,
            black_indices,
            piece_count,
            queen_bucket,
            fake_quantize_acts=True,
            fake_quantize_weights=True,
        )
    assert out.shape == (2, 1)

    perm = np.random.permutation(512 // 2)
    ft_permute_impl(model, perm)
