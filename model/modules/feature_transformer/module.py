import torch
from torch import nn

from .functions import (
    SparseLinearFunction,
    FusedDoubleFtFunction,
    resolve_double_ft_backend,
)
from ..features.composed import ComposedFeatures


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
        merged, bias = self.features.merged_weight_and_bias(
            fake_quantize_weights
        )
        ft_max_act = self.features.quantization.max_ft_activation

        impl = resolve_double_ft_backend(
            us, them, white_indices, black_indices, psqt_indices, merged, bias
        )

        if impl == "fused":
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
            wp = SparseLinearFunction.apply(white_indices, merged, bias, backend=impl)
            bp = SparseLinearFunction.apply(black_indices, merged, bias, backend=impl)

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

