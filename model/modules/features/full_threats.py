import torch
from torch import nn

from .input_feature import InputFeature


class FullThreats(InputFeature):
    HASH = 0x2E6B9D04
    FEATURE_NAME = "Full_Threats"
    INPUT_FEATURE_NAME = "Full_Threats"
    MAX_ACTIVE_FEATURES = 128

    NUM_INPUTS = 59808
    NUM_REAL_FEATURES = 59808
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

    def clip_weights(self, quantization) -> None:
        """Clamp threat weights to quantization-safe range."""
        self.weight.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
