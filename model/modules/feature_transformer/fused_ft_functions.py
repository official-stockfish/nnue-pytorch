import numpy as np
import torch
from torch import autograd

_HAS_CUPY_KERNELS = False
try:
    from .aggregated_ft_kernel import aggregated_ft_backward
    from .fused_ft_kernel import (
        BACKWARD_TILE_SIZE,
        make_fused_double_ft_backward_kernel,
        make_fused_double_ft_forward_kernel,
    )

    _HAS_CUPY_KERNELS = True
except (ImportError, OSError, RuntimeError):
    BACKWARD_TILE_SIZE = 1

# Tiled forward / backward (docs/TILED_FT.md); NNUE_TILED_FORWARD=0 / NNUE_TILED_BACKWARD=0 disable them.
try:
    from . import tiled_ft_fused as _tiled_ft
except (ImportError, OSError, RuntimeError):
    _tiled_ft = None


def _use_aggregated_backward(
    l1_size, batch_size, max_active_features, device, *, mode="auto"
):
    """Select measured defaults; allow other NVIDIA GPUs to be benchmarked."""
    if mode not in ("auto", "scatter", "aggregated"):
        raise ValueError(f"mode must be auto, scatter, or aggregated, got {mode!r}")
    if mode == "scatter" or torch.version.hip is not None:
        return False
    if not (
        512 <= l1_size <= 4096 and l1_size % 128 == 0 and 0 < max_active_features <= 288
    ):
        return False
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        return False
    if mode == "aggregated":
        return True
    # H100 retains its existing threshold. RTX 4090 and A100 measurements show
    # a benefit from 4096 positions; smaller batches keep direct scatter.
    return (
        (capability == (9, 0) and batch_size >= 1024)
        or (capability == (8, 9) and batch_size >= 4096)
        or (capability == (8, 0) and batch_size >= 4096)
    )


def _aggregated_tile_size(l1_size, batch_size, max_active_features, capability):
    """Position-tile size for the aggregation kernel, per measured device.

    A larger tile removes more repeated feature atomics on the production net.
    The tile must stay at or below 16 positions: rows are tracked in a 32-bit
    mask (two rows per position, one per perspective).
    """
    if l1_size == 1024 and max_active_features == 288:
        if capability == (8, 9) and batch_size >= 32768:
            return 14
        if capability == (8, 0) and batch_size >= 8192:
            return 12
        if capability == (9, 0) and batch_size >= 8192:
            return 14
    return 8


class _NoCtx:
    pass


class FusedDoubleFtFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        us,
        them,
        white_indices,
        black_indices,
        weight,
        bias,
        max_ft_activation,
        l1_size,
        act_quant=None,
    ):
        ctx.max_ft_activation = float(max_ft_activation)
        ctx.act_c = None if act_quant is None else float(act_quant[2])
        ctx.l1_size = int(l1_size)

        assert l1_size % 2 == 0

        assert us.is_cuda and them.is_cuda
        assert white_indices.is_cuda and black_indices.is_cuda
        assert weight.is_cuda and bias.is_cuda
        assert (
            us.device
            == them.device
            == white_indices.device
            == black_indices.device
            == weight.device
            == bias.device
        )

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

        keep_acc = _tiled_ft is not None and _tiled_ft.backward_will_run(l1_size)
        tiled = (
            _tiled_ft.forward(
                us, them, white_indices, black_indices, weight, bias,
                max_ft_activation, l1_size, keep_acc=keep_acc, act_quant=act_quant,
            )
            if _tiled_ft is not None
            else None
        )
        ctx.tiled_bwd_prep = (
            _tiled_ft.lookup_backward_prep(white_indices) if _tiled_ft is not None else None
        )
        acc_b = None
        if tiled is not None:
            l0_, clamped_out = tiled
            if keep_acc:  # (acc_w, acc_b): the tiled backward recomputes the clamped values
                clamped_out, acc_b = clamped_out
        else:
            l0_ = torch.empty(batch_size, l1_size, dtype=torch.float32, device=us.device)
            clamped_out = torch.empty(
                batch_size, 4, l1_half, dtype=torch.float32, device=us.device
            )

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
                ),
            )
        ctx.has_acc = acc_b is not None
        ctx.save_for_backward(
            us, them, white_indices, black_indices, weight, bias, clamped_out,
            *((acc_b,) if acc_b is not None else ()),
        )
        if act_quant is not None and not (tiled is not None and _tiled_ft.ACT_FUSED):
            from . import fused_weight_quant as _fwq
            with torch.no_grad():
                l0_ = _fwq._ActFakeQuant.forward(_NoCtx(), l0_, *act_quant)
        return l0_

    @staticmethod
    def backward(ctx, grad_l0):
        us, them, white_indices, black_indices, weight, bias, clamped_out, *acc_b = (
            ctx.saved_tensors
        )
        max_ft_activation = ctx.max_ft_activation
        l1_size = ctx.l1_size

        if ctx.act_c is not None and ctx.act_c != 1.0:
            grad_l0 = grad_l0 * ctx.act_c   # STE of the fused activation fake quantization (as _ActFakeQuant)
        grad_l0 = grad_l0.contiguous()
        if _tiled_ft is not None:
            prep, ctx.tiled_bwd_prep = ctx.tiled_bwd_prep, None
            tiled = _tiled_ft.backward(
                us, them, white_indices, black_indices, grad_l0, clamped_out,
                weight.shape[0], max_ft_activation, l1_size, prep=prep,
                acc=(clamped_out, acc_b[0]) if ctx.has_acc else None,
            )
            if tiled is not None:
                return None, None, None, None, *tiled, None, None, None

        batch_size = white_indices.shape[0]
        max_active_features = white_indices.shape[1]
        output_size = bias.shape[0]

        grad_weight = torch.zeros(
            weight.shape[0], output_size, dtype=torch.float32, device=us.device
        )
        grad_bias = torch.zeros(output_size, dtype=torch.float32, device=us.device)

        # Feature extractors guarantee unique IDs within each perspective.
        # Keep direct scatter for unsupported shapes/devices and small batches.
        if _use_aggregated_backward(
            l1_size, batch_size, max_active_features, us.device
        ):
            tile_size = _aggregated_tile_size(
                l1_size,
                batch_size,
                max_active_features,
                torch.cuda.get_device_capability(us.device),
            )
            aggregated_ft_backward(
                us,
                them,
                white_indices,
                black_indices,
                grad_l0,
                clamped_out,
                grad_weight,
                grad_bias,
                max_ft_activation,
                tile_size=tile_size,
            )
            return None, None, None, None, grad_weight, grad_bias, None, None, None

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
            ),
        )

        return None, None, None, None, grad_weight, grad_bias, None, None, None
