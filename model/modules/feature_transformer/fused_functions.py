import torch
import math
import numpy as np

from torch import autograd

from .fused_kernels import (
    make_fused_nnue_forward_kernel,
    make_fused_nnue_backward_kernel,
)

from .functions import (
    sparse_linear_op,
)

_USE_FUSED_DOUBLE_FT = True

def set_use_fused_double_ft(use_custom: bool):
    global _USE_FUSED_DOUBLE_FT
    _USE_FUSED_DOUBLE_FT = use_custom

def fused_double_ft_op(
    w_indices: torch.Tensor,
    w_values: torch.Tensor,
    b_indices: torch.Tensor,
    b_values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    us: torch.Tensor,
    them: torch.Tensor,
    ft_max_val: float,
    L1: int,
    num_psqt_buckets: int
):
    if _USE_FUSED_DOUBLE_FT:
        return FusedNNUETransformerFunction.apply(
            w_indices, w_values, b_indices, b_values,
            weight, bias, us, them, ft_max_val, L1, num_psqt_buckets
        )

    # --- Unfused Fallback ---

    wp = sparse_linear_op(w_indices, w_values, weight, bias)
    bp = sparse_linear_op(b_indices, b_values, weight, bias)

    w, wpsqt = torch.split(wp, [L1, num_psqt_buckets], dim=1)
    b, bpsqt = torch.split(bp, [L1, num_psqt_buckets], dim=1)

    # Perspective Mixing
    us_view = us.view(-1, 1)
    them_view = them.view(-1, 1)
    l0_ = (us_view * torch.cat([w, b], dim=1)) + (them_view * torch.cat([b, w], dim=1))

    l0_ = torch.clamp(l0_, 0.0, float(ft_max_val))

    l0_s = torch.split(l0_, L1 // 2, dim=1)
    out_l0 = torch.cat([l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]], dim=1)

    return out_l0, wpsqt, bpsqt


class FusedNNUETransformerFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w_indices: torch.Tensor,
        w_values: torch.Tensor,
        b_indices: torch.Tensor,
        b_values: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        us_tensor: torch.Tensor,
        them_tensor: torch.Tensor,
        ft_max_val: float,
        L1: int,
        num_psqt_buckets: int
    ):
        # kernel needs 32 bit float
        ft_max_val = np.float32(ft_max_val)
        # Save tensors needed for rematerialization during the backward pass.
        ctx.save_for_backward(
            w_indices, w_values, b_indices, b_values, weight, bias, us_tensor, them_tensor
        )
        ctx.ft_max_val = ft_max_val
        ctx.L1 = L1
        ctx.num_psqt_buckets = num_psqt_buckets

        batch_size = w_indices.shape[0]
        max_active_indices = w_indices.shape[1]

        # Allocate the exact output topologies
        out_l0 = torch.empty(batch_size, L1, dtype=torch.float32, device=weight.device)
        out_wpsqt = torch.empty(batch_size, num_psqt_buckets, dtype=torch.float32, device=weight.device)
        out_bpsqt = torch.empty(batch_size, num_psqt_buckets, dtype=torch.float32, device=weight.device)

        kernel, num_threads = make_fused_nnue_forward_kernel(max_active_indices, L1, num_psqt_buckets)

        kernel(
            grid=(batch_size,),
            block=(num_threads,),
            args=(
                w_indices.data_ptr(),
                w_values.data_ptr(),
                b_indices.data_ptr(),
                b_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                us_tensor.contiguous().data_ptr(),
                them_tensor.contiguous().data_ptr(),
                out_l0.data_ptr(),
                out_wpsqt.data_ptr(),
                out_bpsqt.data_ptr(),
                ft_max_val
            ),
        )

        return out_l0, out_wpsqt, out_bpsqt

    @staticmethod
    def backward(ctx, grad_out_l0, grad_out_wpsqt, grad_out_bpsqt):
        (
            w_indices, w_values, b_indices, b_values, weight, bias, us_tensor, them_tensor
        ) = ctx.saved_tensors

        # Determine which inputs actually need gradients.
        # Indices match the argument order in forward() (0-indexed)
        # weight is at index 4, bias is at index 5
        needs_weight_grad = ctx.needs_input_grad[4]
        needs_bias_grad = ctx.needs_input_grad[5]

        if not needs_weight_grad and not needs_bias_grad:
            # 11 inputs to forward = 11 Nones returned
            return None, None, None, None, None, None, None, None, None, None, None

        batch_size = w_indices.shape[0]
        max_active_indices = w_indices.shape[1]
        L1 = ctx.L1
        num_psqt_buckets = ctx.num_psqt_buckets
        ft_max_val = ctx.ft_max_val

        weight_grad = torch.zeros_like(weight) if needs_weight_grad else None
        bias_grad = torch.zeros_like(bias) if needs_bias_grad else None

        kernel, threads_per_block_y = make_fused_nnue_backward_kernel(max_active_indices, L1, num_psqt_buckets)

        # A static chunk size of 128 is a safe default for modern hardware,
        # potentially integrate autotune
        chunk_size = 128

        grid_x = math.ceil(batch_size / chunk_size)
        grid_y = math.ceil((L1 // 2 + num_psqt_buckets) / threads_per_block_y)

        kernel(
            grid=(grid_x, grid_y),
            block=(threads_per_block_y,),
            args=(
                w_indices.data_ptr(),
                w_values.data_ptr(),
                b_indices.data_ptr(),
                b_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                us_tensor.contiguous().data_ptr(),
                them_tensor.contiguous().data_ptr(),
                grad_out_l0.contiguous().data_ptr(),
                grad_out_wpsqt.contiguous().data_ptr(),
                grad_out_bpsqt.contiguous().data_ptr(),
                weight_grad.data_ptr() if weight_grad is not None else 0,
                bias_grad.data_ptr() if bias_grad is not None else 0,
                ft_max_val,
                batch_size,
                chunk_size
            ),
        )

        return (
            None, # w_indices
            None, # w_values
            None, # b_indices
            None, # b_values
            weight_grad,
            bias_grad,
            None, # us_tensor (assuming it requires no grad)
            None, # them_tensor (assuming it requires no grad)
            None, # ft_max_val
            None, # L1
            None  # num_psqt_buckets
        )
