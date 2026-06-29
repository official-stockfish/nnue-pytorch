import torch
import torch.nn.functional as F
from torch import autograd
import numpy as np

_HAS_CUPY_KERNELS = False
try:
    from .sparse_linear_kernel import (
        make_sparse_input_linear_forward_kernel,
        make_sparse_input_linear_backward_kernel,
    )
    from .fused_ft_kernel import (
        make_fused_double_ft_forward_kernel,
        make_fused_double_ft_backward_kernel,
    )
    _HAS_CUPY_KERNELS = True
except (ImportError, OSError, RuntimeError):
    pass


def _torch_sparse_linear(feature_indices, weight, bias):
    """Device-agnostic fallback for SparseLinearFunction.

    Computes: output[b] = sum_k(weight[indices[b,k]]) + bias
    Negative entries in feature_indices are treated as padding and
    contribute nothing to the sum. Uses F.embedding_bag for memory efficiency.
    """
    batch_size, max_active = feature_indices.shape
    mask = feature_indices >= 0

    if feature_indices.device.type == "mps":
        safe_indices = feature_indices.clamp(min=0).long().reshape(-1)
        per_sample_weights = mask.to(weight.dtype).reshape(-1, 1)
        gathered_weight = F.embedding(safe_indices, weight)
        output = (gathered_weight * per_sample_weights).reshape(
            batch_size, max_active, weight.shape[1]
        ).sum(dim=1)
        return output + bias

    safe_indices = feature_indices.clamp(min=0).long().reshape(-1)
    per_sample_weights = mask.to(weight.dtype).reshape(-1)
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
        ctx.save_for_backward(feature_indices, weight, bias)

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
        )

        kernel = make_sparse_input_linear_forward_kernel(
            max_active_features, output_size
        )
        kernel(
            grid=(batch_size,),
            args=(
                feature_indices.data_ptr(),
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

        feature_indices, weight, bias = ctx.saved_tensors

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
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr(),
            ),
        )

        return None, weight_grad, bias_grad


# Allowed modes: "auto", "fused", "sparse", "torch"
_DOUBLE_FT_IMPL = "auto"


def set_double_ft_impl(mode: str):
    """Set the implementation mode for the double feature transformer.

    Allowed modes: "auto", "fused", "sparse", "torch"
    """
    global _DOUBLE_FT_IMPL
    if mode not in ("auto", "fused", "sparse", "torch"):
        raise ValueError(f"Invalid mode: {mode}")
    _DOUBLE_FT_IMPL = mode


def get_double_ft_impl() -> str:
    return _DOUBLE_FT_IMPL


def resolve_double_ft_backend(
    us: torch.Tensor,
    them: torch.Tensor,
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    psqt_indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> str:
    """Resolves the implementation mode, ensuring strict enforcement and no silent fallbacks."""
    mode = get_double_ft_impl()

    cupy_available = _HAS_CUPY_KERNELS
    all_cuda = (
        us.is_cuda
        and them.is_cuda
        and white_indices.is_cuda
        and black_indices.is_cuda
        and psqt_indices.is_cuda
        and weight.is_cuda
        and bias.is_cuda
    )
    cuda_capable = cupy_available and all_cuda

    if mode == "fused":
        if not cupy_available:
            raise RuntimeError("Fused double FT backend requested, but CuPy kernels are not available.")
        if not all_cuda:
            raise RuntimeError("Fused double FT backend requested, but not all tensors/parameters are on CUDA.")
        return "fused"

    elif mode == "sparse":
        if not cupy_available:
            raise RuntimeError("Sparse backend requested, but CuPy kernels are not available.")
        if not all_cuda:
            raise RuntimeError("Sparse backend requested, but not all tensors/parameters are on CUDA.")
        return "sparse"

    elif mode == "torch":
        return "torch"

    elif mode == "auto":
        return "fused" if cuda_capable else "torch"

    else:
        raise ValueError(f"Invalid double FT implementation mode: {mode}")


class SparseLinearFunction:
    """
    Uses custom CuPy CUDA kernel when available. Otherwise falls back to a
    PyTorch implementation that works on any device (CPU, MPS).
    """
    @staticmethod
    def apply(feature_indices, weight, bias, backend: str = "auto"):
        if backend == "auto":
            if _HAS_CUPY_KERNELS and feature_indices.is_cuda and weight.is_cuda and bias.is_cuda:
                return _CudaSparseLinearFunction.apply(feature_indices, weight, bias)
            return _torch_sparse_linear(feature_indices, weight, bias)

        elif backend == "sparse":
            if not _HAS_CUPY_KERNELS:
                raise RuntimeError("CuPy sparse linear kernel is not available.")
            if not (feature_indices.is_cuda and weight.is_cuda and bias.is_cuda):
                raise RuntimeError("Sparse CUDA kernel requested but tensors are not on CUDA.")
            return _CudaSparseLinearFunction.apply(feature_indices, weight, bias)

        elif backend == "torch":
            return _torch_sparse_linear(feature_indices, weight, bias)

        else:
            raise ValueError(f"Invalid SparseLinear backend requested: {backend}")


class _CudaFusedDoubleFtFunction(autograd.Function):
    @staticmethod
    def forward(ctx, us, them, white_indices, black_indices, psqt_indices, weight, bias, max_ft_activation, l1_size):
        ctx.save_for_backward(us, them, white_indices, black_indices, psqt_indices, weight, bias)
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

        l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
        wpsqt = torch.empty(batch_size, 1, dtype=torch.float32, device=us.device)
        bpsqt = torch.empty(batch_size, 1, dtype=torch.float32, device=us.device)

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
                np.int32(output_size),
            )
        )

        return l0_, wpsqt, bpsqt

    @staticmethod
    def backward(ctx, grad_l0, grad_wpsqt, grad_bpsqt):
        us, them, white_indices, black_indices, psqt_indices, weight, bias = ctx.saved_tensors
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
                grad_l0.data_ptr(),
                grad_wpsqt.data_ptr(),
                grad_bpsqt.data_ptr(),
                grad_weight.data_ptr(),
                grad_bias.data_ptr(),
                np.int32(output_size),
            )
        )

        return None, None, None, None, None, grad_weight, grad_bias, None, None


class FusedDoubleFtFunction:
    @staticmethod
    def apply(us, them, white_indices, black_indices, psqt_indices, weight, bias, max_ft_activation, l1_size):
        return _CudaFusedDoubleFtFunction.apply(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            weight,
            bias,
            max_ft_activation,
            l1_size,
        )
