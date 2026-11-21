"""Fused FT activation fake quantization against the QuantizationManager reference it replaces
(bit-identical, gradient included)."""

import pytest
import torch

from model.cupy_utils import HAS_CUPY
from model.quantize import QuantizationConfig, QuantizationManager


@pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_CUPY), reason="needs CUDA + CuPy"
)
@pytest.mark.parametrize("correction", [None, 0.75])
def test_fused_act_path_matches_quantization_manager(correction):
    qm = QuantizationManager(QuantizationConfig())
    if correction is not None:
        qm.l0_correction_factor = correction
    x = (torch.rand(257, 1024, device="cuda") * 1.1).requires_grad_()
    out = qm.fake_quantize_ft_act_and_correct(x)
    ref = qm.fake_quantize_ft_act(x) * qm.l0_correction_factor
    assert torch.equal(out, ref)
    g = torch.randn_like(out)
    assert torch.equal(
        torch.autograd.grad(out, x, g)[0], torch.autograd.grad(ref, x, g)[0]
    )
