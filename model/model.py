import torch
from torch import nn

from .config import ModelConfig
from .features import FeatureSet
from .modules import DoubleFeatureTransformerSlice, LayerStacks
from .quantize import QuantizationConfig, QuantizationManager


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_set: FeatureSet,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        self.input = DoubleFeatureTransformerSlice(
            feature_set.num_features, self.L1 + self.num_psqt_buckets
        )
        self.feature_set = feature_set
        self.layer_stacks = LayerStacks(self.num_ls_buckets, config)

        self.quantization = QuantizationManager(quantize_config)
        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

        self._init_layers()

    def _init_layers(self):
        self._zero_virtual_feature_weights()
        self._init_psqt()

    def _zero_virtual_feature_weights(self):
        """
        We zero all virtual feature weights because there's not need for them
        to be initialized; they only aid the training of correlated features.
        """
        weights = self.input.weight
        with torch.no_grad():
            for a, b in self.feature_set.get_virtual_feature_ranges():
                weights[a:b, :] = 0.0
        self.input.weight = nn.Parameter(weights)

    def _init_psqt(self):
        input_weights = self.input.weight
        input_bias = self.input.bias
        # 1.0 / kPonanzaConstant
        scale = 1 / self.quantization.nnue2score

        with torch.no_grad():
            initial_values = self.feature_set.get_initial_psqt_features()
            assert len(initial_values) == self.feature_set.num_features

            new_weights = (
                torch.tensor(
                    initial_values,
                    device=input_weights.device,
                    dtype=input_weights.dtype,
                )
                * scale
            )

            for i in range(self.num_psqt_buckets):
                input_weights[:, self.L1 + i] = new_weights
                # Bias doesn't matter because it cancels out during
                # inference during perspective averaging. We set it to 0
                # just for the sake of it. It might still diverge away from 0
                # due to gradient imprecision but it won't change anything.
                input_bias[self.L1 + i] = 0.0

        self.input.weight = nn.Parameter(input_weights)
        self.input.bias = nn.Parameter(input_bias)

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
                            min_weight_t = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                            p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
                        if max_weight is not None:
                            max_weight_t = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                            p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
                    else:
                        if min_weight is not None and max_weight is not None:
                            p_data_fp32.clamp_(min_weight, max_weight)
                        else:
                            raise Exception("Not supported.")
                    p.data.copy_(p_data_fp32)
    
    def clip_threat_weights(self):
        p = self.input.weight[0:79856]
        p_data_fp32 = p.data
        min_weight = -128 / 255
        max_weight = 127 / 255
        p_data_fp32.clamp_(min_weight, max_weight)
        p.data.copy_(p_data_fp32)

    def set_feature_set(self, new_feature_set: FeatureSet):
        """
        This method attempts to convert the model from using the self.feature_set
        to new_feature_set. Currently only works for adding virtual features.
        """
        if self.feature_set.name == new_feature_set.name:
            return

        # TODO: Implement this for more complicated conversions.
        #       Currently we support only a single feature block.
        if len(self.feature_set.features) > 1:
            raise Exception(
                "Cannot change feature set from {} to {}.".format(
                    self.feature_set.name, new_feature_set.name
                )
            )

        # Currently we only support conversion for feature sets with
        # one feature block each so we'll dig the feature blocks directly
        # and forget about the set.
        old_feature_block = self.feature_set.features[0]
        new_feature_block = new_feature_set.features[0]

        # next(iter(new_feature_block.factors)) is the way to get the
        # first item in a OrderedDict. (the ordered dict being str : int
        # mapping of the factor name to its size).
        # It is our new_feature_factor_name.
        # For example old_feature_block.name == "HalfKP"
        # and new_feature_factor_name == "HalfKP^"
        # We assume here that the "^" denotes factorized feature block
        # and we would like feature block implementers to follow this convention.
        # So if our current feature_set matches the first factor in the new_feature_set
        # we only have to add the virtual feature on top of the already existing real ones.
        if old_feature_block.name == next(iter(new_feature_block.factors)):
            # We can just extend with zeros since it's unfactorized -> factorized
            weights = self.input.weight
            padding = weights.new_zeros(
                (new_feature_block.num_virtual_features, weights.shape[1])
            )
            weights = torch.cat([weights, padding], dim=0)
            self.input.weight = nn.Parameter(weights)
            self.feature_set = new_feature_set
        else:
            raise Exception(
                "Cannot change feature set from {} to {}.".format(
                    self.feature_set.name, new_feature_set.name
                )
            )

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
