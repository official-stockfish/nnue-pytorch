import torch
from torch import nn

from .config import ModelConfig
from .modules import LayerStacks, ComposedFeatures, DoubleFeatureTransformer, get_feature_cls
from .quantize import QuantizationManager

class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        feature_cls = get_feature_cls(feature_name)
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        features = ComposedFeatures(feature_cls, self.L1, self.num_psqt_buckets, self.quantization)
        self.input = DoubleFeatureTransformer(features)
        self.feature_name = self.input.features.FEATURE_NAME
        self.input_feature_name = self.input.features.INPUT_FEATURE_NAME
        self.feature_hash = self.input.features.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config, self.quantization)

        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.features.init_weights()


    @torch.no_grad()
    def clip_weights(self, include_input):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        if include_input:
            self.input.features.clip_weights(self.quantization)

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


    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.input.features.zero_virtual_weights()
        self.layer_stacks.zero_virtual_weights()


    def forward_ft(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        psqt_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )

    def calculate_buckets(self, piece_count: torch.Tensor):
        psqt_indices = (piece_count - 1) // 4
        layer_stack_indices = psqt_indices

        return psqt_indices, layer_stack_indices


    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        piece_count: torch.Tensor,
        fake_quantize_acts: bool=True,
        fake_quantize_weights: bool=True,
    ):
        psqt_indices, layer_stack_indices = self.calculate_buckets(piece_count)

        l0_, wpsqt, bpsqt = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )
        # The PSQT values are averaged over perspectives. "Their" perspective
        # has a negative influence (us-0.5 is 0.5 for white and -0.5 for black,
        # which does both the averaging and sign flip for black to move)
        x = self.layer_stacks(l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights) + (wpsqt - bpsqt) * (us - 0.5)

        return x
