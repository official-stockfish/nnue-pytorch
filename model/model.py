from typing import Generator

import torch
from torch import nn, Tensor

from .config import ModelConfig
from .feature_transformer import DoubleFeatureTransformerSlice
from .features import FeatureSet


class LayerStacks(nn.Module):
    def __init__(self, count: int, config: ModelConfig):
        super().__init__()

        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.count = count
        self.l1 = nn.Linear(2 * self.L1 // 2, (self.L2 + 1) * count)
        # Factorizer only for the first layer because later
        # there's a non-linearity and factorization breaks.
        # This is by design. The weights in the further layers should be
        # able to diverge a lot.
        self.l1_fact = nn.Linear(2 * self.L1 // 2, self.L2 + 1, bias=True)
        self.l2 = nn.Linear(self.L2 * 2, self.L3 * count)
        self.output = nn.Linear(self.L3, 1 * count)

        self._init_layers()

    def _init_layers(self):
        l1_weight = self.l1.weight
        l1_bias = self.l1.bias
        l1_fact_weight = self.l1_fact.weight
        l1_fact_bias = self.l1_fact.bias
        l2_weight = self.l2.weight
        l2_bias = self.l2.bias
        output_weight = self.output.weight
        output_bias = self.output.bias

        with torch.no_grad():
            l1_fact_weight.fill_(0.0)
            l1_fact_bias.fill_(0.0)
            output_bias.fill_(0.0)

            for i in range(1, self.count):
                # Force all layer stacks to be initialized in the same way.
                l1_weight[i * (self.L2 + 1) : (i + 1) * (self.L2 + 1), :] = l1_weight[
                    0 : (self.L2 + 1), :
                ]
                l1_bias[i * (self.L2 + 1) : (i + 1) * (self.L2 + 1)] = l1_bias[
                    0 : (self.L2 + 1)
                ]
                l2_weight[i * self.L3 : (i + 1) * self.L3, :] = l2_weight[
                    0 : self.L3, :
                ]
                l2_bias[i * self.L3 : (i + 1) * self.L3] = l2_bias[0 : self.L3]
                output_weight[i : i + 1, :] = output_weight[0:1, :]

        self.l1.weight = nn.Parameter(l1_weight)
        self.l1.bias = nn.Parameter(l1_bias)
        self.l1_fact.weight = nn.Parameter(l1_fact_weight)
        self.l1_fact.bias = nn.Parameter(l1_fact_bias)
        self.l2.weight = nn.Parameter(l2_weight)
        self.l2.bias = nn.Parameter(l2_bias)
        self.output.weight = nn.Parameter(output_weight)
        self.output.bias = nn.Parameter(output_bias)

    def forward(self, x: Tensor, ls_indices: Tensor):
        idx_offset = torch.arange(
            0, x.shape[0] * self.count, self.count, device=x.device
        )

        indices = ls_indices.flatten() + idx_offset

        l1s_ = self.l1(x).reshape((-1, self.count, self.L2 + 1))
        l1f_ = self.l1_fact(x)
        # https://stackoverflow.com/questions/55881002/pytorch-tensor-indexing-how-to-gather-rows-by-tensor-containing-indices
        # basically we present it as a list of individual results and pick not only based on
        # the ls index but also based on batch (they are combined into one index)
        l1c_ = l1s_.view(-1, self.L2 + 1)[indices]
        l1c_, l1c_out = l1c_.split(self.L2, dim=1)
        l1f_, l1f_out = l1f_.split(self.L2, dim=1)
        l1x_ = l1c_ + l1f_
        # multiply sqr crelu result by (127/128) to match quantized version
        l1x_ = torch.clamp(
            torch.cat([torch.pow(l1x_, 2.0) * (127 / 128), l1x_], dim=1), 0.0, 1.0
        )

        l2s_ = self.l2(l1x_).reshape((-1, self.count, self.L3))
        l2c_ = l2s_.view(-1, self.L3)[indices]
        l2x_ = torch.clamp(l2c_, 0.0, 1.0)

        l3s_ = self.output(l2x_).reshape((-1, self.count, 1))
        l3c_ = l3s_.view(-1, 1)[indices]
        l3x_ = l3c_ + l1f_out + l1c_out

        return l3x_

    def get_coalesced_layer_stacks(self) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        # During training the buckets are represented by a single, wider, layer.
        # This representation needs to be transformed into individual layers
        # for the serializer, because the buckets are interpreted as separate layers.
        for i in range(self.count):
            with torch.no_grad():
                l1 = nn.Linear(2 * self.L1 // 2, self.L2 + 1)
                l2 = nn.Linear(self.L2 * 2, self.L3)
                output = nn.Linear(self.L3, 1)
                l1.weight.data = (
                    self.l1.weight[i * (self.L2 + 1) : (i + 1) * (self.L2 + 1), :]
                    + self.l1_fact.weight.data
                )
                l1.bias.data = (
                    self.l1.bias[i * (self.L2 + 1) : (i + 1) * (self.L2 + 1)]
                    + self.l1_fact.bias.data
                )
                l2.weight.data = self.l2.weight[i * self.L3 : (i + 1) * self.L3, :]
                l2.bias.data = self.l2.bias[i * self.L3 : (i + 1) * self.L3]
                output.weight.data = self.output.weight[i : (i + 1), :]
                output.bias.data = self.output.bias[i : (i + 1)]
                yield l1, l2, output


class NNUEModel(nn.Module):
    def __init__(
        self,
        feature_set: FeatureSet,
        config: ModelConfig,
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

        self.nnue2score = 600.0
        self.weight_scale_hidden = 64.0
        self.weight_scale_out = 16.0
        self.quantized_one = 127.0

        max_hidden_weight = self.quantized_one / self.weight_scale_hidden
        max_out_weight = (self.quantized_one * self.quantized_one) / (
            self.nnue2score * self.weight_scale_out
        )
        self.weight_clipping = [
            {
                "params": [self.layer_stacks.l1.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
                "virtual_params": self.layer_stacks.l1_fact.weight,
            },
            {
                "params": [self.layer_stacks.l2.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
            },
            {
                "params": [self.layer_stacks.output.weight],
                "min_weight": -max_out_weight,
                "max_weight": max_out_weight,
            },
        ]

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
        scale = 1 / self.nnue2score

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
        us: Tensor,
        them: Tensor,
        white_indices: Tensor,
        white_values: Tensor,
        black_indices: Tensor,
        black_values: Tensor,
        psqt_indices: Tensor,
        layer_stack_indices: Tensor,
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
