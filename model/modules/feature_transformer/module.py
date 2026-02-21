import math

import torch
import torch.nn.functional as F
from torch import nn

from .functions import SparseLinearFunction


class BaseFeatureTransformer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.weight = nn.Parameter(
            torch.empty((num_inputs, num_outputs), dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.empty(num_outputs, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        sigma = math.sqrt(1 / self.num_inputs)
        with torch.no_grad():
            self.weight.uniform_(-sigma, sigma)
            self.bias.uniform_(-sigma, sigma)

    def expand_input_layer(self, additional_features):
        assert additional_features >= 0
        if additional_features == 0:
            return

        with torch.no_grad():
            new_weight = F.pad(self.weight, (0, 0, 0, additional_features), value=0)

            self.weight = nn.Parameter(new_weight)
            self.num_inputs += additional_features


class FeatureTransformer(BaseFeatureTransformer):
    def forward(self, feature_indices, feature_values):
        return SparseLinearFunction.apply(
            feature_indices, feature_values, self.weight, self.bias
        )


class DoubleFeatureTransformer(BaseFeatureTransformer):
    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        return (
            SparseLinearFunction.apply(
                feature_indices_0,
                feature_values_0,
                self.weight,
                self.bias,
            ),
            SparseLinearFunction.apply(
                feature_indices_1,
                feature_values_1,
                self.weight,
                self.bias,
            ),
        )
