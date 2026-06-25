import torch
from torch import nn

from .functions import SparseLinearFunction, FusedDoubleFtFunction, _HAS_CUPY_KERNELS
from ..features.composed import ComposedFeatures

_USE_FUSED_DOUBLE_FT = True


def get_use_fused_double_ft() -> bool:
    return _USE_FUSED_DOUBLE_FT


def set_use_fused_double_ft(val: bool):
    global _USE_FUSED_DOUBLE_FT
    _USE_FUSED_DOUBLE_FT = val


class DoubleFeatureTransformer(nn.Module):
    def __init__(self, features: ComposedFeatures):
        super().__init__()
        self.features = features

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        psqt_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            get_use_fused_double_ft()
            and _HAS_CUPY_KERNELS
            and us.is_cuda
            and them.is_cuda
            and white_indices.is_cuda
            and black_indices.is_cuda
            and psqt_indices.is_cuda
        ):
            merged, bias = self.features.merged_weight_and_bias(
                fake_quantize_weights
            )
            ft_max_act = self.features.quantization.max_ft_activation
            l0_, wpsqt, bpsqt = FusedDoubleFtFunction.apply(
                us,
                them,
                white_indices,
                black_indices,
                psqt_indices,
                merged,
                bias,
                ft_max_act,
                self.features.l1_size,
            )
        else:
            merged, bias = self.features.merged_weight_and_bias(
                fake_quantize_weights
            )
            wp = SparseLinearFunction.apply(white_indices, merged, bias)
            bp = SparseLinearFunction.apply(black_indices, merged, bias)

            w, wpsqt = torch.split(wp, self.features.l1_size, dim=1)
            b, bpsqt = torch.split(bp, self.features.l1_size, dim=1)

            psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
            wpsqt = wpsqt.gather(1, psqt_indices_unsq)
            bpsqt = bpsqt.gather(1, psqt_indices_unsq)

            l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
            # do not fake quantize sum of (quantized) weights
            l0_ = self.features.quantization.clip_ft_act(l0_)

            l0_s = torch.split(l0_, self.features.l1_size // 2, dim=1)
            l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
            l0_ = torch.cat(l0_s1, dim=1)

        if fake_quantize_acts:
            l0_ = self.features.quantization.fake_quantize_ft_act(l0_)
        # We multiply by a correction factor,
        # so we can use only bitshift and multiplication at inference.
        # When using fake quantization any correction factor
        # not equal 1.0 will lead to diverging discrete grids
        l0_ = l0_ * self.features.quantization.l0_correction_factor

        return l0_, wpsqt, bpsqt

