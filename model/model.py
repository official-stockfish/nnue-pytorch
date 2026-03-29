import torch
from torch import nn

from .config import ModelConfig
from .modules import LayerStacks, get_feature_cls
from .quantize import QuantizationConfig, QuantizationManager

_HAS_METAL_FUSED = False
try:
    from .modules.feature_transformer.metal import (
        is_available as _metal_is_available,
        metal_fused_double_forward_l0,
        metal_fused_composed_double_forward_l0,
    )
    _HAS_METAL_FUSED = _metal_is_available()
except (ImportError, ModuleNotFoundError):
    pass


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        feature_cls = get_feature_cls(feature_name)
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        self.input = feature_cls(self.L1 + self.num_psqt_buckets)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config)

        self.quantization = QuantizationManager(quantize_config)
        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights(num_psqt_buckets, self.quantization.nnue2score)

    @torch.no_grad()
    def clip_weights(self):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        vp_r, vp_c = virtual_params.shape
                        xs = p_data.shape[0] // vp_r
                        ys = p_data.shape[1] // vp_c
                        p_view = p_data.view(xs, vp_r, ys, vp_c)
                        vp_bcast = virtual_params.view(1, vp_r, 1, vp_c)
                        min_bound = (min_weight - vp_bcast) if min_weight is not None else None
                        max_bound = (max_weight - vp_bcast) if max_weight is not None else None
                        p_view.clamp_(min_bound, max_bound)
                    else:
                        p_data.clamp_(min_weight, max_weight)

    def clip_input_weights(self):
        self.input.clip_weights(self.quantization)

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        white_values: torch.Tensor,
        black_indices: torch.Tensor,
        black_values: torch.Tensor,
        psqt_indices: torch.Tensor,
        layer_stack_indices: torch.Tensor,
    ):
        if _HAS_METAL_FUSED and white_indices.device.type == "mps":
            ft = self.input
            if (hasattr(ft, "features") and len(ft.features) == 2
                    and hasattr(ft.features[1], "virtual_weight")):
                fa, fb = ft.features
                l0_, wpsqt, bpsqt = metal_fused_composed_double_forward_l0(
                    white_indices, white_values, black_indices, black_values,
                    fa.weight, fb.weight, fb.virtual_weight, ft.bias,
                    us, them, self.L1, self.num_psqt_buckets,
                    fa.NUM_INPUTS, fb.NUM_INPUTS_VIRTUAL,
                )
            elif hasattr(ft, "weight"):
                weight = ft.weight
                l0_, wpsqt, bpsqt = metal_fused_double_forward_l0(
                    white_indices, white_values, black_indices, black_values,
                    weight, ft.bias, us, them, self.L1, self.num_psqt_buckets,
                )
            else:
                weight = torch.cat(
                    [f.merged_weight() for f in ft.features], dim=0
                )
                l0_, wpsqt, bpsqt = metal_fused_double_forward_l0(
                    white_indices, white_values, black_indices, black_values,
                    weight, ft.bias, us, them, self.L1, self.num_psqt_buckets,
                )
        else:
            wp, bp = self.input(
                white_indices, white_values, black_indices, black_values
            )
            w, wpsqt = torch.split(wp, self.L1, dim=1)
            b, bpsqt = torch.split(bp, self.L1, dim=1)
            l0_ = (us * torch.cat([w, b], dim=1)) + (
                them * torch.cat([b, w], dim=1)
            )
            l0_ = torch.clamp(l0_, 0.0, 1.0)

            l0_s = torch.split(l0_, self.L1 // 2, dim=1)
            l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
            l0_ = torch.cat(l0_s1, dim=1) * (127 / 128)

        psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
        wpsqt = wpsqt.gather(1, psqt_indices_unsq)
        bpsqt = bpsqt.gather(1, psqt_indices_unsq)
        # The PSQT values are averaged over perspectives. "Their" perspective
        # has a negative influence (us-0.5 is 0.5 for white and -0.5 for black,
        # which does both the averaging and sign flip for black to move)
        x = self.layer_stacks(l0_, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)

        return x
