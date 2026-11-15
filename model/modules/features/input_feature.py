import math
from abc import ABC, abstractmethod

import torch
from torch import nn


class InputFeature(nn.Module, ABC):
    NAME: str
    HASH: int
    NUM_INPUTS: int
    MAX_ACTIVE_FEATURES: int
    NUM_REAL_FEATURES: int
    EXPORT_WEIGHT_DTYPE: torch.dtype = torch.int16
    # The rows form NUM_BUCKETS consecutive blocks of NUM_INPUTS // NUM_BUCKETS rows (K32H3: one per
    # king bucket and state), and the active rows of one perspective all fall in a single block.
    NUM_BUCKETS: int = 1

    @abstractmethod
    def merged_weight(self) -> torch.Tensor:
        """Return effective weight matrix (with virtual weights merged if applicable)."""

    def virtual_layout(self) -> tuple[torch.Tensor, int] | None:
        """(virtual_weight, num_buckets) when merged_weight() is
        ``weight + virtual_weight.repeat(num_buckets, 1)``, else None. Fused kernels use it to build
        the merged weight without merged_weight(); keep the two in sync."""
        return None

    @abstractmethod
    def coalesce(self) -> None: ...

    @abstractmethod
    def zero_virtual_weights(self) -> None: ...

    @abstractmethod
    def init_weights(self) -> None: ...

    @abstractmethod
    def get_export_weights(self) -> torch.Tensor: ...

    @abstractmethod
    def load_export_weights(self, export_weight: torch.Tensor) -> None: ...

    def clip_weights(self, quantization) -> None:
        min_w, max_w = quantization.get_ft_weight_bounds(self.EXPORT_WEIGHT_DTYPE)
        self.weight.data.clamp_(min_w, max_w)

    def reset_parameters(self):
        sigma = math.sqrt(1 / self.NUM_INPUTS)
        with torch.no_grad():
            self.weight.uniform_(-sigma, sigma)
