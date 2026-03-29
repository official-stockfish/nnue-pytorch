"""Custom Metal kernels for the sparse feature transformer on Apple Silicon.

Provides a hand-tuned Metal equivalent of the CuPy CUDA kernels in kernel.py,
using the same threadgroup-per-batch-element strategy with per-thread output
slicing.

Build the extension with:
    python setup_metal.py build_ext --inplace
"""

import os
import sys

import torch
from torch import autograd

_dir = os.path.dirname(__file__)
_cpp = None
_shader_sources: dict[str, str] = {}


def _get_shader(name: str) -> str:
    if name not in _shader_sources:
        with open(os.path.join(_dir, name)) as f:
            _shader_sources[name] = f.read()
    return _shader_sources[name]


def _load_extension():
    """Try to load the Metal extension (AOT-built or JIT-compiled)."""
    global _cpp
    if _cpp is not None:
        return _cpp

    if sys.platform != "darwin":
        return None

    # 1) Try importing the AOT-built module (from setup_metal.py build_ext -i).
    try:
        import sparse_linear_metal_cpp as mod  # noqa: F811

        _cpp = mod
        return _cpp
    except ImportError:
        pass

    # 2) Fall back to JIT compilation via torch.utils.cpp_extension.load().
    #    The .mm source is compiled as Objective-C++ by passing the language
    #    flag explicitly and renaming to .cpp for the build system.
    try:
        import shutil
        import tempfile

        from torch.utils.cpp_extension import load

        tmp_dir = tempfile.mkdtemp(prefix="metal_sparse_")
        cpp_path = os.path.join(tmp_dir, "sparse_linear.cpp")
        shutil.copy2(os.path.join(_dir, "sparse_linear.mm"), cpp_path)

        _cpp = load(
            name="sparse_linear_metal_cpp",
            sources=[cpp_path],
            extra_cflags=["-std=c++17", "-x", "objective-c++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=False,
        )
        return _cpp
    except Exception:
        return None


def is_available() -> bool:
    """True when Metal kernels can be used (macOS + MPS + extension builds)."""
    if sys.platform != "darwin":
        return False
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    try:
        return _load_extension() is not None
    except Exception:
        return False


class MetalSparseLinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias):
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)

        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32
        assert weight.dtype == torch.float32
        assert bias.dtype == torch.float32

        return _cpp.sparse_linear_forward(
            feature_indices,
            feature_values,
            weight,
            bias,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )

    @staticmethod
    def backward(ctx, grad_output):
        feature_indices, feature_values, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        weight_grad = _cpp.sparse_linear_backward(
            feature_indices,
            feature_values,
            grad_output,
            weight.size(0),
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )
        bias_grad = grad_output.sum(dim=0)
        return None, None, weight_grad, bias_grad


def metal_sparse_linear(feature_indices, feature_values, weight, bias):
    """Drop-in replacement for sparse_linear when tensors are on MPS."""
    return MetalSparseLinearFunction.apply(
        feature_indices, feature_values, weight, bias
    )


class DoubleMetalSparseLinearFunction(autograd.Function):
    """Both perspectives in one autograd node — shares a single weight_grad
    tensor in the backward, eliminating the ~0.84 ms autograd accumulation
    overhead from adding two separate 96 MB gradient tensors."""

    @staticmethod
    def forward(ctx, w_indices, w_values, b_indices, b_values, weight, bias):
        ctx.save_for_backward(w_indices, w_values, b_indices, b_values, weight, bias)
        wp, bp = _cpp.sparse_linear_double_forward(
            w_indices, w_values, b_indices, b_values, weight, bias,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )
        return wp, bp

    @staticmethod
    def backward(ctx, grad_wp, grad_bp):
        w_indices, w_values, b_indices, b_values, weight, bias = ctx.saved_tensors
        grad_wp = grad_wp.contiguous()
        grad_bp = grad_bp.contiguous()
        weight_grad = _cpp.sparse_linear_double_backward(
            w_indices, w_values, b_indices, b_values,
            grad_wp, grad_bp, weight.size(0),
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )
        bias_grad = grad_wp.sum(dim=0) + grad_bp.sum(dim=0)
        return None, None, None, None, weight_grad, bias_grad


def metal_double_sparse_linear(w_indices, w_values, b_indices, b_values, weight, bias):
    """Double-perspective sparse linear with shared weight_grad in backward."""
    return DoubleMetalSparseLinearFunction.apply(
        w_indices, w_values, b_indices, b_values, weight, bias
    )


class FusedDoubleForwardL0Function(autograd.Function):
    """Single autograd node for: double sparse_linear → L0 mixing.

    Eliminates autograd overhead for the intermediate wp/bp tensors and
    uses a single shared weight_grad in the backward (saves ~0.84 ms per
    step from avoiding the redundant alloc + element-wise addition)."""

    @staticmethod
    def forward(ctx, w_idx, w_val, b_idx, b_val, weight, bias, us, them, L1, psqt):
        wp, bp = _cpp.sparse_linear_double_forward(
            w_idx, w_val, b_idx, b_val, weight, bias,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )
        l0, wpsqt, bpsqt = _cpp.l0_mixing_forward(
            wp, bp, us, them, L1, psqt,
            _get_shader("l0_mixing.metal"),
        )
        ctx.save_for_backward(w_idx, w_val, b_idx, b_val, us, wp, bp)
        ctx.num_inputs = weight.size(0)
        return l0, wpsqt, bpsqt

    @staticmethod
    def backward(ctx, grad_l0, grad_wpsqt, grad_bpsqt):
        w_idx, w_val, b_idx, b_val, us, wp, bp = ctx.saved_tensors
        them = 1.0 - us

        weight_grad, bias_grad = _cpp.fused_backward(
            grad_l0.contiguous(), grad_wpsqt.contiguous(), grad_bpsqt.contiguous(),
            wp, bp, us, them,
            w_idx, w_val, b_idx, b_val,
            ctx.num_inputs,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
            _get_shader("l0_mixing.metal"),
        )
        return None, None, None, None, weight_grad, bias_grad, None, None, None, None


def metal_fused_double_forward_l0(w_idx, w_val, b_idx, b_val, weight, bias,
                                   us, them, L1, psqt):
    """Fused double sparse_linear + L0 mixing — single autograd node."""
    return FusedDoubleForwardL0Function.apply(
        w_idx, w_val, b_idx, b_val, weight, bias, us, them, L1, psqt
    )


class FusedL0MixingFunction(autograd.Function):
    @staticmethod
    def forward(ctx, wp, bp, us, them, L1, psqt):
        ctx.save_for_backward(wp, bp, us, them)
        l0, wpsqt, bpsqt = _cpp.l0_mixing_forward(
            wp, bp, us, them, L1, psqt,
            _get_shader("l0_mixing.metal"),
        )
        return l0, wpsqt, bpsqt

    @staticmethod
    def backward(ctx, grad_l0, grad_wpsqt, grad_bpsqt):
        wp, bp, us, them = ctx.saved_tensors
        grad_wp, grad_bp = _cpp.l0_mixing_backward(
            grad_l0.contiguous(),
            grad_wpsqt.contiguous(),
            grad_bpsqt.contiguous(),
            wp, bp, us, them,
            _get_shader("l0_mixing.metal"),
        )
        return grad_wp, grad_bp, None, None, None, None


def metal_l0_mixing(wp, bp, us, them, L1, psqt):
    """Fused L0 mixing: replaces ~7 separate MPS ops with one Metal kernel."""
    return FusedL0MixingFunction.apply(wp, bp, us, them, L1, psqt)


class FusedLossFunction(autograd.Function):
    """Fuses ~20 element-wise loss ops into two Metal kernels (fwd + bwd)."""

    @staticmethod
    def forward(ctx, scorenet, score, outcome,
                in_offset, in_scaling, out_offset, out_scaling,
                actual_lambda, pow_exp, qp_asymmetry, w1_factor, w2):
        partial_wloss, partial_weights = _cpp.loss_forward(
            scorenet, score, outcome,
            in_offset, in_scaling, out_offset, out_scaling,
            actual_lambda, pow_exp, qp_asymmetry, w1_factor, w2,
            _get_shader("loss.metal"),
        )
        weights_sum = partial_weights.sum()
        loss = partial_wloss.sum() / weights_sum

        ctx.save_for_backward(scorenet, score, outcome, weights_sum)
        ctx.loss_scalars = (
            in_offset, in_scaling, out_offset, out_scaling,
            actual_lambda, pow_exp, qp_asymmetry, w1_factor, w2,
        )
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        scorenet, score, outcome, weights_sum = ctx.saved_tensors
        (in_offset, in_scaling, out_offset, out_scaling,
         actual_lambda, pow_exp, qp_asymmetry, w1_factor, w2) = ctx.loss_scalars

        grad_scale = (grad_output / weights_sum).view(1)

        grad_scorenet = _cpp.loss_backward(
            scorenet, score, outcome, grad_scale,
            in_offset, in_scaling, out_offset, out_scaling,
            actual_lambda, pow_exp, qp_asymmetry, w1_factor, w2,
            _get_shader("loss.metal"),
        )
        return grad_scorenet.view_as(scorenet), *([None] * 11)


def metal_fused_loss(scorenet, score, outcome,
                     in_offset, in_scaling, out_offset, out_scaling,
                     actual_lambda, pow_exp, qp_asymmetry, w1, w2):
    """Fused loss: replaces ~20 separate MPS dispatches."""
    w1_factor = 2.0 ** w1 - 1.0
    return FusedLossFunction.apply(
        scorenet, score, outcome,
        in_offset, in_scaling, out_offset, out_scaling,
        actual_lambda, pow_exp, qp_asymmetry, w1_factor, w2,
    )
