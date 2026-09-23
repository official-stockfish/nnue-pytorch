import numpy as np
import torch
from torch import autograd

_HAS_CUPY_KERNELS = False
try:
    from .fused_ft_kernel import (
        BACKWARD_TILE_SIZE,
        make_fused_double_ft_backward_kernel,
        make_fused_double_ft_forward_kernel,
    )
    _HAS_CUPY_KERNELS = True
except (ImportError, OSError, RuntimeError):
    BACKWARD_TILE_SIZE = 1


class FusedDoubleFtFunction(autograd.Function):
    @staticmethod
    def forward(ctx, us, them, white_indices, black_indices, weight, bias, max_ft_activation, l1_size):
        ctx.max_ft_activation = float(max_ft_activation)
        ctx.l1_size = int(l1_size)

        assert l1_size % 2 == 0

        assert us.is_cuda and them.is_cuda
        assert white_indices.is_cuda and black_indices.is_cuda
        assert weight.is_cuda and bias.is_cuda
        assert us.device == them.device == white_indices.device == black_indices.device == weight.device == bias.device

        assert us.dtype == torch.float32 and them.dtype == torch.float32
        assert white_indices.dtype == torch.int32 and black_indices.dtype == torch.int32
        assert weight.dtype == torch.float32 and bias.dtype == torch.float32

        assert white_indices.ndim == 2 and black_indices.ndim == 2
        assert len(weight.shape) == 2
        assert len(bias.shape) == 1
        assert weight.shape[1] == bias.shape[0]
        assert white_indices.shape == black_indices.shape

        assert us.is_contiguous() and them.is_contiguous()
        assert white_indices.is_contiguous() and black_indices.is_contiguous()
        assert weight.is_contiguous() and bias.is_contiguous()

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        l1_half = l1_size // 2

        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
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
                weight.data_ptr(),
                bias.data_ptr(),
                np.float32(max_ft_activation),
                l0_.data_ptr(),
                clamped_out.data_ptr(),
                np.int32(output_size),
            )
        )

        ctx.save_for_backward(us, them, white_indices, black_indices, weight, bias, clamped_out)
        return l0_

    @staticmethod
    def backward(ctx, grad_l0):
        us, them, white_indices, black_indices, weight, bias, clamped_out = ctx.saved_tensors
        max_ft_activation = ctx.max_ft_activation
        l1_size = ctx.l1_size

        grad_l0 = grad_l0.contiguous()

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
                weight.data_ptr(),
                bias.data_ptr(),
                np.float32(max_ft_activation),
                grad_l0.data_ptr(),
                clamped_out.data_ptr(),
                grad_weight.data_ptr(),
                grad_bias.data_ptr(),
                np.int32(batch_size),
                np.int32(output_size),
            )
        )

        return None, None, None, None, grad_weight, grad_bias, None, None
