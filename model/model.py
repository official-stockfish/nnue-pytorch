import torch
from torch import nn

from .config import ModelConfig
from .modules import ComposedFeatureTransformer, LayerStacks, get_feature_cls
from .modules.layer_stacks import stack_count, stack_keys
from .quantize import QuantizationManager


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
    ):
        super().__init__()

        feature_cls = get_feature_cls(feature_name)
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)

        # The data loader computes the layer-stack index from these keys
        self.ls_keys = config.ls_keys
        self.num_ls_buckets = stack_count(stack_keys(config.ls_keys))

        self.input = ComposedFeatureTransformer(feature_cls, self.L1, self.quantization)
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config, self.quantization)

        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self.input.init_weights()

    @torch.no_grad()
    def clip_weights(self, include_input):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        if include_input:
            self.input.clip_weights(self.quantization)

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
                    if "stacked_layer" in group:
                        ge = group["stacked_layer"].group_expansion()
                        if ge is not None:
                            gw = ge[0].detach()
                            if not torch.is_tensor(min_weight):
                                min_weight = p_data_fp32.new_full(
                                    p_data_fp32.shape, min_weight
                                )
                            if not torch.is_tensor(max_weight):
                                max_weight = p_data_fp32.new_full(
                                    p_data_fp32.shape, max_weight
                                )
                            min_weight = min_weight - gw
                            max_weight = max_weight - gw
                    p_data_fp32.clamp_(min_weight, max_weight)

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.input.zero_virtual_weights()
        self.layer_stacks.zero_virtual_weights()

    def forward_ft(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
    ) -> torch.Tensor:
        return self.input(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        layer_stack_indices: torch.Tensor,
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ):
        l0_ = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )
        x = self.layer_stacks(
            l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights
        )

        return x
