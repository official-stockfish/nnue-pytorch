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
    def get_ft_params(self) -> list[nn.Parameter]:
        """Return parameters belonging to the Feature Transformer group."""

    @abstractmethod
    def get_pqst_params(self) -> list[nn.Parameter]:
        """Return parameters belonging to the PSQT group."""

    @abstractmethod
    def reset_parameters(self) -> None:
        """Initialize parameters."""

    @abstractmethod
    def merged_weight(self) -> torch.Tensor:
        """Return effective weight matrix (with virtual weights merged if applicable)."""

    @abstractmethod
    def coalesce(self) -> None: ...

    @abstractmethod
    def init_weights(self, nnue2score: float) -> None: ...

    @abstractmethod
    def get_export_weights(self) -> torch.Tensor: ...

    @abstractmethod
    def load_export_weights(self, export_weight: torch.Tensor) -> None: ...

    @abstractmethod
    def reset_parameters(self): ...

    def clip_weights(self, quantization) -> None:
        pass
