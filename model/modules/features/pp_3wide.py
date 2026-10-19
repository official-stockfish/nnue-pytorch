import torch
from torch import nn

from .input_feature import InputFeature


class PP3Wide(InputFeature):
    HASH = 0x65F06AEB
    FEATURE_NAME = "PP_3Wide^"
    INPUT_FEATURE_NAME = "PP_3Wide"
    MAX_ACTIVE_FEATURES = 128

    NUM_PLANES = 2163
    NUM_BUCKETS = 8
    NUM_INPUTS = NUM_PLANES * NUM_BUCKETS
    NUM_INPUTS_VIRTUAL = NUM_PLANES
    NUM_REAL_FEATURES = NUM_INPUTS
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )
        self.virtual_weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.virtual_weight.repeat(self.NUM_BUCKETS, 1)

    @torch.no_grad()
    def coalesce(self) -> None:
        self.weight.add_(self.virtual_weight.repeat(self.NUM_BUCKETS, 1))
        self.zero_virtual_weights()

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.virtual_weight.zero_()

    @torch.no_grad()
    def init_weights(self) -> None:
        self.zero_virtual_weights()

    def clip_weights(self, quantization) -> None:
        min_weight, max_weight = quantization.get_ft_weight_bounds(
            self.EXPORT_WEIGHT_DTYPE
        )
        weights = self.weight.data.view(self.NUM_BUCKETS, self.NUM_PLANES, -1)
        virtual = self.virtual_weight.data.view(1, self.NUM_PLANES, -1)

        # Only the merged weights are exported, so clip their sum to the i8 range.
        weights.add_(virtual)
        weights.clamp_(min_weight, max_weight)
        weights.sub_(virtual)

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.merged_weight()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)
        self.zero_virtual_weights()
