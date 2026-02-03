import torch
from torch import nn


from ..feature_transformer import DoubleFeatureTransformer, SparseLinearFunction


class HalfKav2Hm(DoubleFeatureTransformer):
    def __init__(self, num_outputs: int):
        self.NUM_SQ: int = 64
        self.NUM_PT: int = 12
        self.NUM_PLANES: int = self.NUM_SQ * self.NUM_PT
        self.NUM_BUCKETS: int = self.NUM_SQ // 2
        self.NUM_INPUTS: int = self.NUM_PLANES * self.NUM_BUCKETS
        self.NUM_INPUTS_VIRTUAL: int = self.NUM_PLANES

        self.virtual_weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_outputs, dtype=torch.float32)
        )

        super().__init__(self.NUM_INPUTS, num_outputs)

    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        self.merged_weight = self.weight + self.virtual_weight.repeat(
            self.NUM_BUCKETS, 1
        )
        return (
            SparseLinearFunction.apply(
                feature_indices_0,
                feature_values_0,
                self.merged_weight,
                self.bias,
            ),
            SparseLinearFunction.apply(
                feature_indices_1,
                feature_values_1,
                self.merged_weight,
                self.bias,
            ),
        )

    @torch.no_grad
    def coalesce(self) -> None:
        self.weight._add(self.virtual_weight.repeat(self.NUM_BUCKETS, 1))
        self.virtual_weight._zero(
            self.NUM_INPUTS_VIRTUAL, self.num_outputs, dtype=torch.float32
        )
