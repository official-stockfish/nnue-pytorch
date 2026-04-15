import torch
import torch.nn.functional as F
from torch import autograd

from ...metal_support import (
    MPS_AVAILABLE,
    metal_sparse_linear,
    metal_double_sparse_linear,
)

_HAS_CUPY_KERNELS = False
try:
    from .kernel import (
        make_sparse_input_linear_forward_kernel,
        make_sparse_input_linear_backward_kernel,
    )

    _HAS_CUPY_KERNELS = True
except (ImportError, ModuleNotFoundError):
    pass


def sparse_linear(feature_indices, weight, bias):
    """Sparse linear: output[b] = sum_k(weight[indices[b,k]]) + bias.

    Feature values are always 1.0 and have been removed from the interface.
    Dispatches to a hand-tuned CuPy CUDA kernel when available, a custom Metal
    kernel on MPS, or falls back to a pure-PyTorch implementation.
    """
    if _HAS_CUPY_KERNELS and feature_indices.is_cuda:
        return _CudaSparseLinearFunction.apply(
            feature_indices, weight, bias
        )
    if MPS_AVAILABLE and feature_indices.device.type == "mps":
        return metal_sparse_linear(feature_indices, weight, bias)
    return _torch_sparse_linear(feature_indices, weight, bias)


def double_sparse_linear(
    w_indices, b_indices, weight, bias
):
    """Both perspectives in one call — saves ~0.84 ms on MPS by sharing
    the weight_grad allocation in the backward pass."""
    if MPS_AVAILABLE and w_indices.device.type == "mps":
        return metal_double_sparse_linear(
            w_indices, b_indices, weight, bias
        )
    return (
        sparse_linear(w_indices, weight, bias),
        sparse_linear(b_indices, weight, bias),
    )


def _torch_sparse_linear(feature_indices, weight, bias):
    """Pure PyTorch implementation using F.embedding_bag for memory efficiency.

    Instead of materialising the full (batch, max_active, output_size) gathered
    tensor, embedding_bag fuses the lookup, per-sample weighting, and reduction
    into a single call with O(batch * output_size) memory.
    """
    batch_size, max_active = feature_indices.shape

    mask = feature_indices >= 0
    safe_indices = feature_indices.clamp(min=0).long().reshape(-1)
    per_sample_weights = mask.float().reshape(-1)

    offsets = torch.arange(
        0,
        batch_size * max_active,
        max_active,
        device=feature_indices.device,
    )

    output = F.embedding_bag(
        safe_indices,
        weight,
        offsets,
        mode="sum",
        per_sample_weights=per_sample_weights,
    )
    return output + bias


class _CudaSparseLinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, feature_indices, weight, bias):
        feature_values = torch.ones_like(feature_indices, dtype=torch.float32)
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)

        assert len(feature_indices.shape) == 2
        assert feature_indices.dtype == torch.int32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(
            batch_size,
            output_size,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        kernel = make_sparse_input_linear_forward_kernel(
            max_active_features, output_size
        )
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr(),
            ),
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert not ctx.needs_input_grad[0]

        grad_output = grad_output.contiguous()

        feature_indices, feature_values, weight, bias = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(
            weight.shape[0], weight.shape[1], dtype=torch.float32, device=device
        )
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        kernel = make_sparse_input_linear_backward_kernel(
            max_active_features, output_size
        )
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr(),
            ),
        )

        return None, weight_grad, bias_grad


# Legacy alias
SparseLinearFunction = _CudaSparseLinearFunction
