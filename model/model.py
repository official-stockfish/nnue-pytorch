import torch
from torch import nn

from .config import ModelConfig
from .modules import LayerStacks
from .modules.features import get_feature_cls
from .quantize import QuantizationConfig, QuantizationManager


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
        self.feature_name = feature_cls.FEATURE_NAME
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        self.input = feature_cls(self.L1 + self.num_psqt_buckets)
        self.feature_hash = feature_cls.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config)

        self.quantization = QuantizationManager(quantize_config)
        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights(num_psqt_buckets, self.quantization.nnue2score)

    def clip_weights(self):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data_fp32 = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                        ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                        expanded_virtual_layer = virtual_params.repeat(xs, ys)
                        if min_weight is not None:
                            min_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                        if max_weight is not None:
                            max_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                    p_data_fp32.clamp_(min_weight, max_weight)

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
        wp, bp = self.input(white_indices, white_values, black_indices, black_values)
        w, wpsqt = torch.split(wp, self.L1, dim=1)
        b, bpsqt = torch.split(bp, self.L1, dim=1)
        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        l0_ = torch.clamp(l0_, 0.0, 1.0)

        l0_s = torch.split(l0_, self.L1 // 2, dim=1)
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
        # We multiply by 127/128 because in the quantized network 1.0 is represented by 127
        # and it's more efficient to divide by 128 instead.
        l0_ = torch.cat(l0_s1, dim=1) * (127 / 128)

        psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
        wpsqt = wpsqt.gather(1, psqt_indices_unsq)
        bpsqt = bpsqt.gather(1, psqt_indices_unsq)
        # The PSQT values are averaged over perspectives. "Their" perspective
        # has a negative influence (us-0.5 is 0.5 for white and -0.5 for black,
        # which does both the averaging and sign flip for black to move)
        x = self.layer_stacks(l0_, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)

        return x
