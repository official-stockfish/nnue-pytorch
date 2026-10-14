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

    @abstractmethod
    def merged_weight(self) -> torch.Tensor:
        """Return effective weight matrix (with virtual weights merged if applicable)."""

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
