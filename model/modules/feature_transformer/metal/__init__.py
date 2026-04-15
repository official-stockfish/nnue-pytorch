"""Custom Metal kernels for the sparse feature transformer on Apple Silicon.

Provides a hand-tuned Metal equivalent of the CuPy CUDA kernels in kernel.py,
using the same threadgroup-per-batch-element strategy with per-thread output
slicing.

Build the extension with:
    python setup_metal.py build_ext --inplace

Hot-path classes (used during MPS fused training):
    FusedComposedDoubleForwardL0Function  -- composed FT forward + L0 mixing
    FusedDoubleForwardL0Function          -- single-weight FT forward + L0 mixing
    FusedLossFunction                     -- fused loss forward + backward
    IndexedStackedLinearFunction          -- layer stack indexed GEMM
    FusedSqrCReluFunction                 -- fused l1 activation
    metal_fused_adam_step_multi           -- fused optimizer step

Test / fallback classes (not on the default fused training path):
    MetalSparseLinearFunction             -- single-perspective sparse linear
    DoubleMetalSparseLinearFunction       -- double-perspective sparse linear
    FusedL0MixingFunction                 -- standalone L0 mixing
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
    """Single-perspective sparse linear. Not on the fused training path --
    used by tests and ComposedFeatureTransformer.forward (CPU/CUDA fallback)."""

    @staticmethod
    def forward(ctx, feature_indices, weight, bias):
        ctx.save_for_backward(feature_indices, weight, bias)

        assert feature_indices.dtype == torch.int32
        assert weight.dtype == torch.float32
        assert bias.dtype == torch.float32

        return _cpp.sparse_linear_forward(
            feature_indices,
            weight,
            bias,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )

    @staticmethod
    def backward(ctx, grad_output):
        feature_indices, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        weight_grad = _cpp.sparse_linear_backward(
            feature_indices,
            grad_output,
            weight.size(0),
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )
        bias_grad = grad_output.sum(dim=0)
        return None, weight_grad, bias_grad


def metal_sparse_linear(feature_indices, weight, bias):
    """Drop-in replacement for sparse_linear when tensors are on MPS."""
    return MetalSparseLinearFunction.apply(
        feature_indices, weight, bias
    )


class DoubleMetalSparseLinearFunction(autograd.Function):
    """Both perspectives in one autograd node. Not on the fused training path
    -- superseded by FusedDoubleForwardL0Function which also fuses L0 mixing.
    Kept for DoubleFeatureTransformer compatibility and tests."""

    @staticmethod
    def forward(ctx, w_indices, b_indices, weight, bias):
        ctx.save_for_backward(w_indices, b_indices, weight, bias)

        wp, bp = _cpp.sparse_linear_double_forward(
            w_indices, b_indices, weight, bias,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )
        return wp, bp

    @staticmethod
    def backward(ctx, grad_wp, grad_bp):
        w_indices, b_indices, weight, bias = ctx.saved_tensors
        grad_wp = grad_wp.contiguous()
        grad_bp = grad_bp.contiguous()

        weight_grad = _cpp.sparse_linear_double_backward(
            w_indices, b_indices,
            grad_wp, grad_bp, weight.size(0),
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
        )

        bias_grad = _cpp.bias_grad_sum(
            grad_wp, grad_bp, _get_shader("l0_mixing.metal"),
        )
        return None, None, weight_grad, bias_grad


def metal_double_sparse_linear(w_indices, b_indices, weight, bias):
    """Double-perspective sparse linear with shared weight_grad in backward."""
    return DoubleMetalSparseLinearFunction.apply(
        w_indices, b_indices, weight, bias
    )


class FusedDoubleForwardL0Function(autograd.Function):
    """Single autograd node for: double sparse_linear → L0 mixing.

    Eliminates autograd overhead for the intermediate wp/bp tensors and
    uses a single shared weight_grad in the backward (saves ~0.84 ms per
    step from avoiding the redundant alloc + element-wise addition)."""

    @staticmethod
    def forward(ctx, w_idx, b_idx, weight, bias, us, them, L1, psqt):
        l0, wpsqt, bpsqt, wp, bp = _cpp.sparse_linear_double_forward_l0(
            w_idx, b_idx, weight, bias,
            us, them, L1, psqt,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
            _get_shader("l0_mixing.metal"),
        )
        ctx.save_for_backward(w_idx, b_idx, us, wp, bp)
        ctx.num_inputs = weight.size(0)
        return l0, wpsqt, bpsqt

    @staticmethod
    def backward(ctx, grad_l0, grad_wpsqt, grad_bpsqt):
        w_idx, b_idx, us, wp, bp = ctx.saved_tensors
        them = 1.0 - us

        weight_grad, bias_grad = _cpp.fused_backward(
            grad_l0.contiguous(), grad_wpsqt.contiguous(), grad_bpsqt.contiguous(),
            wp, bp, us, them,
            w_idx, b_idx,
            ctx.num_inputs,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
            _get_shader("l0_mixing.metal"),
        )
        return None, None, weight_grad, bias_grad, None, None, None, None


def metal_fused_double_forward_l0(w_idx, b_idx, weight, bias,
                                   us, them, L1, psqt):
    """Fused double sparse_linear + L0 mixing — single autograd node."""
    return FusedDoubleForwardL0Function.apply(
        w_idx, b_idx, weight, bias, us, them, L1, psqt
    )


class FusedComposedDoubleForwardL0Function(autograd.Function):
    """Like FusedDoubleForwardL0Function but reads from component weight
    buffers directly, eliminating the merged weight tensor allocation.

    Accepts weight_a (first feature), weight_b (second feature base),
    virtual_w (second feature virtual weight), and distributes gradients
    to the component parameters in the backward pass."""

    @staticmethod
    def forward(ctx, w_idx, b_idx,
                weight_a, weight_b, virtual_w, bias,
                us, them, L1, psqt, vw_period):
        l0, wpsqt, bpsqt, wp, bp = _cpp.sparse_linear_composed_double_forward_l0(
            w_idx, b_idx,
            weight_a, weight_b, virtual_w, bias,
            vw_period, us, them, L1, psqt,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
            _get_shader("l0_mixing.metal"),
        )
        ctx.save_for_backward(w_idx, b_idx, us, wp, bp)
        ctx.num_inputs = weight_a.size(0) + weight_b.size(0)
        ctx.boundary = weight_a.size(0)
        ctx.vw_period = vw_period
        return l0, wpsqt, bpsqt

    @staticmethod
    def backward(ctx, grad_l0, grad_wpsqt, grad_bpsqt):
        w_idx, b_idx, us, wp, bp = ctx.saved_tensors
        them = 1.0 - us

        weight_grad, bias_grad = _cpp.fused_backward(
            grad_l0.contiguous(), grad_wpsqt.contiguous(), grad_bpsqt.contiguous(),
            wp, bp, us, them,
            w_idx, b_idx,
            ctx.num_inputs,
            _get_shader("sparse_linear.metal"),
            _get_shader("sparse_linear_backward_cas.metal"),
            _get_shader("sparse_linear_backward_native.metal"),
            _get_shader("l0_mixing.metal"),
        )

        bnd = ctx.boundary
        vw_p = ctx.vw_period
        grad_weight_a = weight_grad[:bnd]
        grad_weight_b = weight_grad[bnd:]
        grad_virtual_w = grad_weight_b.view(-1, vw_p, weight_grad.size(1)).sum(0)

        return (None, None,
                grad_weight_a, grad_weight_b, grad_virtual_w, bias_grad,
                None, None, None, None, None)


def metal_fused_composed_double_forward_l0(
        w_idx, b_idx,
        weight_a, weight_b, virtual_w, bias,
        us, them, L1, psqt, vw_period):
    """Composed fused double sparse_linear + L0 mixing — no merged weight."""
    return FusedComposedDoubleForwardL0Function.apply(
        w_idx, b_idx,
        weight_a, weight_b, virtual_w, bias,
        us, them, L1, psqt, vw_period,
    )


class FusedL0MixingFunction(autograd.Function):
    """Standalone L0 mixing. Not on the fused training path -- L0 is fused
    inside FusedDoubleForwardL0Function / FusedComposedDoubleForwardL0Function.
    Kept for modular use and tests."""

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


class IndexedStackedLinearFunction(autograd.Function):
    """Forward uses a Metal kernel that computes only the selected output
    block per batch element (count-x less work). Backward grad_x uses Metal;
    grad_weight/bias falls back to PyTorch matmul (faster than a per-element
    Metal reduction for the large l1 layer dimensions)."""

    @staticmethod
    def forward(ctx, x, weight, bias, indices, out_size, count):
        output = _cpp.indexed_stacked_linear_forward(
            x, weight, bias, indices, out_size,
            _get_shader("stacked_linear.metal"),
        )
        ctx.save_for_backward(x, weight, indices)
        ctx.out_size = out_size
        ctx.count = count
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, indices = ctx.saved_tensors
        out_size = ctx.out_size
        count = ctx.count
        grad_output = grad_output.contiguous()

        grad_x = _cpp.indexed_stacked_linear_backward_x(
            grad_output, weight, indices,
            _get_shader("stacked_linear.metal"),
        )

        batch = grad_output.size(0)
        total_out = count * out_size
        grad_full = grad_output.new_zeros(batch, total_out)
        idx_expanded = (indices.view(-1, 1) * out_size
                        + torch.arange(out_size, device=grad_output.device))
        grad_full.scatter_(1, idx_expanded, grad_output)

        grad_weight = grad_full.t() @ x
        grad_bias = grad_full.sum(0)

        return grad_x, grad_weight, grad_bias, None, None, None


def metal_indexed_stacked_linear(x, weight, bias, indices, out_size, count):
    """Indexed stacked linear — Metal kernel computes only the selected
    output block, avoiding count-x wasted compute."""
    return IndexedStackedLinearFunction.apply(
        x, weight, bias, indices, out_size, count
    )


class FusedSqrCReluFunction(autograd.Function):
    """Fuses split → pow(2) * (255/256) → cat → clamp(0,1) into one Metal
    kernel for the l1 activation in layer stacks."""

    @staticmethod
    def forward(ctx, l1c, L2):

        l1x, l1x_out = _cpp.sqr_crelu_forward(
            l1c, L2, _get_shader("stacked_linear.metal"),
        )
        ctx.save_for_backward(l1c)
        ctx.L2 = L2
        return l1x, l1x_out

    @staticmethod
    def backward(ctx, grad_l1x, grad_l1x_out):
        l1c, = ctx.saved_tensors

        grad_l1c = _cpp.sqr_crelu_backward(
            grad_l1x.contiguous(), grad_l1x_out.contiguous(),
            l1c, ctx.L2, _get_shader("stacked_linear.metal"),
        )
        return grad_l1c, None


def metal_sqr_crelu(l1c, L2):
    """Fused squared-clamp-relu activation for layer stack l1."""
    return FusedSqrCReluFunction.apply(l1c, L2)


def metal_fused_adam_step_multi(params, grads, grad_mas, variance_mas,
                                beta1_sq, one_minus_beta1_sq,
                                beta2, one_minus_beta2,
                                inv_sqrt_bc2, step_size, eps):
    """Multi-tensor fused Adam step — one C++ call for all params."""
    _cpp.fused_adam_step_multi(
        params, grads, grad_mas, variance_mas,
        beta1_sq, one_minus_beta1_sq,
        beta2, one_minus_beta2,
        inv_sqrt_bc2, step_size, eps,
        _get_shader("optimizer.metal"),
    )
