import math

import torch
from torch import nn

from .functions import SparseLinearFunction


class BaseFeatureTransformer(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        sigma = math.sqrt(1 / num_inputs)

        self.weight = nn.Parameter(
            torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma)
            - sigma
        )
        self.bias = nn.Parameter(
            torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma
        )


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
