import torch
from torch import autograd
import numpy as np

_HAS_CUPY_KERNELS = False
try:
    from .fused_ft_kernel import (
        make_fused_double_ft_forward_kernel,
        make_fused_double_ft_backward_kernel,
        BACKWARD_TILE_SIZE,
    )
    _HAS_CUPY_KERNELS = True
except (ImportError, OSError, RuntimeError):
    BACKWARD_TILE_SIZE = 1
    pass


class FusedDoubleFtFunction(autograd.Function):
    @staticmethod
    def forward(ctx, us, them, white_indices, black_indices, psqt_indices, weight, bias, max_ft_activation, l1_size):
        ctx.max_ft_activation = float(max_ft_activation)
        ctx.l1_size = int(l1_size)

        assert l1_size % 2 == 0

        assert us.is_cuda and them.is_cuda
        assert white_indices.is_cuda and black_indices.is_cuda and psqt_indices.is_cuda
        assert weight.is_cuda and bias.is_cuda
        assert us.device == them.device == white_indices.device == black_indices.device == psqt_indices.device == weight.device == bias.device

        assert us.dtype == torch.float32 and them.dtype == torch.float32
        assert white_indices.dtype == torch.int32 and black_indices.dtype == torch.int32
        assert psqt_indices.dtype == torch.int64
        assert weight.dtype == torch.float32 and bias.dtype == torch.float32

        assert white_indices.ndim == 2 and black_indices.ndim == 2
        assert psqt_indices.ndim == 1
        assert len(weight.shape) == 2
        assert len(bias.shape) == 1
        assert weight.shape[1] == bias.shape[0]
        assert white_indices.shape == black_indices.shape
        assert white_indices.shape[0] == psqt_indices.shape[0]

        assert us.is_contiguous() and them.is_contiguous()
        assert white_indices.is_contiguous() and black_indices.is_contiguous() and psqt_indices.is_contiguous()
        assert weight.is_contiguous() and bias.is_contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        l1_half = l1_size // 2

        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
        wpsqt = torch.empty(batch_size, 1, dtype=torch.float32, device=us.device)
        bpsqt = torch.empty(batch_size, 1, dtype=torch.float32, device=us.device)
        clamped_out = torch.empty(batch_size, 4, l1_half, dtype=torch.float32, device=us.device)

        output_size = bias.shape[0]
        kernel = make_fused_double_ft_forward_kernel(max_active_features, l1_size)
        kernel(
            grid=(batch_size,),
            args=(
                us.data_ptr(),
                them.data_ptr(),
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                psqt_indices.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                np.float32(max_ft_activation),
                l0_.data_ptr(),
                wpsqt.data_ptr(),
                bpsqt.data_ptr(),
                clamped_out.data_ptr(),
                np.int32(output_size),
            )
        )

        ctx.save_for_backward(us, them, white_indices, black_indices, psqt_indices, weight, bias, clamped_out)
        return l0_, wpsqt, bpsqt

    @staticmethod
    def backward(ctx, grad_l0, grad_wpsqt, grad_bpsqt):
        us, them, white_indices, black_indices, psqt_indices, weight, bias, clamped_out = ctx.saved_tensors
        max_ft_activation = ctx.max_ft_activation
        l1_size = ctx.l1_size

        grad_l0 = grad_l0.contiguous()
        grad_wpsqt = grad_wpsqt.contiguous()
        grad_bpsqt = grad_bpsqt.contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        output_size = bias.shape[0]

        grad_weight = torch.zeros(weight.shape[0], output_size, dtype=torch.float32, device=us.device)
        grad_bias = torch.zeros(output_size, dtype=torch.float32, device=us.device)

        kernel = make_fused_double_ft_backward_kernel(max_active_features, l1_size)
        grid_size = (batch_size + BACKWARD_TILE_SIZE - 1) // BACKWARD_TILE_SIZE
        kernel(
            grid=(grid_size,),
            args=(
                us.data_ptr(),
                them.data_ptr(),
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                psqt_indices.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                np.float32(max_ft_activation),
                grad_l0.data_ptr(),
                grad_wpsqt.data_ptr(),
                grad_bpsqt.data_ptr(),
                clamped_out.data_ptr(),
                grad_weight.data_ptr(),
                grad_bias.data_ptr(),
                np.int32(batch_size),
                np.int32(output_size),
            )
        )

        return None, None, None, None, None, grad_weight, grad_bias, None, None
