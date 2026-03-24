import math
import torch
from torch import nn

from .input_feature import InputFeature


class FullThreats(InputFeature):
    HASH = 0x8F234CB8
    FEATURE_NAME = "Full_Threats"
    INPUT_FEATURE_NAME = "Full_Threats"
    MAX_ACTIVE_FEATURES = 128

    NUM_INPUTS = 60144
    NUM_REAL_FEATURES = 60144
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, l1: int, num_psqt_buckets: int):
        super().__init__()

        self.l1 = l1
        self.num_psqt_buckets = num_psqt_buckets
        self.num_outputs = l1 + num_psqt_buckets

        self.weight_ft = nn.Parameter(
            torch.empty(self.NUM_INPUTS, l1, dtype=torch.float32)
        )
        self.weight_psqt = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_psqt_buckets, dtype=torch.float32)
        )

        self.reset_parameters()

    def get_ft_params(self) -> list[nn.Parameter]:
        return [self.weight_ft]

    def get_psqt_params(self) -> list[nn.Parameter]:
        return [self.weight_psqt]

    @torch.no_grad()
    def reset_parameters(self) -> None:
        sigma = math.sqrt(1 / self.NUM_INPUTS)
        self.weight_ft.uniform_(-sigma, sigma)
        self.weight_psqt.uniform_(-sigma, sigma)

    def merged_weight(self) -> torch.Tensor:
        return torch.cat([self.weight_ft, self.weight_psqt], dim=1)

    @torch.no_grad()
    def coalesce(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self, nnue2score: float) -> None:
        """Threats have no piece values, so PSQT columns are zero."""
        self.weight_psqt.zero_()

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return torch.cat([self.weight_ft.data, self.weight_psqt.data], dim=1)

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight_ft.data.copy_(export_weight[:, :self.l1])
        self.weight_psqt.data.copy_(export_weight[:, self.l1:])

    def clip_weights(self, quantization) -> None:
        """Clamp threat weights to quantization-safe range."""
        self.weight_ft.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
        self.weight_psqt.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
