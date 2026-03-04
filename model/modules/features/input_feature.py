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
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None: ...

    @abstractmethod
    def get_export_weights(self) -> torch.Tensor: ...

    @abstractmethod
    def load_export_weights(self, export_weight: torch.Tensor) -> None: ...

    def clip_weights(self, quantization) -> None:
        pass

    def reset_parameters(self):
        sigma = math.sqrt(1 / self.NUM_INPUTS)
        with torch.no_grad():
            self.weight.uniform_(-sigma, sigma)
