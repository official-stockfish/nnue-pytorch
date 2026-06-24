import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.functions import FusedDoubleFtFunction
from model.modules.feature_transformer.functions import _HAS_CUPY_KERNELS

@pytest.mark.skipif(not torch.cuda.is_available() or not _HAS_CUPY_KERNELS, reason="CUDA and CuPy required for custom kernel")
def test_fused_double_ft():
    batch_size = 4
    max_active = 32
    num_inputs = 100
    l1 = 32
    num_psqt_buckets = 8
    
    output_size = l1 + num_psqt_buckets
    
    us = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    them = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    
    # ensure non-negative indices and some padding (-1)
    white_indices = torch.randint(0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda")
    white_indices[:, -2:] = -1
    black_indices = torch.randint(0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda")
    black_indices[:, -2:] = -1
    
    psqt_indices = torch.randint(0, num_psqt_buckets, (batch_size,), dtype=torch.int64, device="cuda")
    
    weight = torch.randn(num_inputs, output_size, dtype=torch.float32, device="cuda", requires_grad=True)
    bias = torch.randn(output_size, dtype=torch.float32, device="cuda", requires_grad=True)
    
    max_ft_act = 127.0
    
    # 1) Fused kernel
    l0_fused, wpsqt_fused, bpsqt_fused = FusedDoubleFtFunction.apply(
        us, them, white_indices, black_indices, psqt_indices, weight, bias, max_ft_act, l1
    )
    
    loss_fused = l0_fused.sum() + wpsqt_fused.sum() + bpsqt_fused.sum()
    loss_fused.backward()
    
    grad_weight_fused = weight.grad.clone()
    grad_bias_fused = bias.grad.clone()
    
    # 2) Fallback
    weight.grad.zero_()
    bias.grad.zero_()
    
    from model.modules.feature_transformer.functions import _torch_sparse_linear
    
    wp = _torch_sparse_linear(white_indices, weight, bias)
    bp = _torch_sparse_linear(black_indices, weight, bias)
    
    w, wpsqt_all = torch.split(wp, l1, dim=1)
    b, bpsqt_all = torch.split(bp, l1, dim=1)
    
    psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
    wpsqt = wpsqt_all.gather(1, psqt_indices_unsq)
    bpsqt = bpsqt_all.gather(1, psqt_indices_unsq)
    
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    l0_ = torch.clamp(l0_, 0.0, max_ft_act)
    
    l0_s = torch.split(l0_, l1 // 2, dim=1)
    l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
    l0_fallback = torch.cat(l0_s1, dim=1)
    
    loss_fallback = l0_fallback.sum() + wpsqt.sum() + bpsqt.sum()
    loss_fallback.backward()
    
    # Compare
    torch.testing.assert_close(l0_fused, l0_fallback, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(wpsqt_fused, wpsqt, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(bpsqt_fused, bpsqt, atol=1e-5, rtol=1e-4)
    
    torch.testing.assert_close(grad_weight_fused, weight.grad, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(grad_bias_fused, bias.grad, atol=1e-4, rtol=1e-3)
