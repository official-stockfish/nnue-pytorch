"""Global feature contract shared with Stockfish's experimental NNUE format.

Supports have virtual positive/negative input rows but only one real weight
table. The other inputs retain unrestricted embeddings for both perspectives.
"""

import torch
from torch import nn

from .input_feature import InputFeature


class _GlobalFeature(InputFeature):
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.weight = nn.Parameter(torch.empty(self.NUM_REAL_FEATURES, num_outputs))
        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass

    @torch.no_grad()
    def init_weights(self) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.detach().clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.copy_(export_weight)


class GlobalSupports(_GlobalFeature):
    HASH = 0x47535031
    FEATURE_NAME = INPUT_FEATURE_NAME = "Global_Supports"
    NUM_REAL_FEATURES = 14952
    NUM_INPUTS = 2 * NUM_REAL_FEATURES
    MAX_ACTIVE_FEATURES = 256

    def merged_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, -self.weight), dim=0)


class GlobalAttacks(_GlobalFeature):
    HASH = 0x47415431
    FEATURE_NAME = INPUT_FEATURE_NAME = "Global_Attacks"
    NUM_INPUTS = NUM_REAL_FEATURES = 29904
    MAX_ACTIVE_FEATURES = 256


class PawnFrontierPairs(_GlobalFeature):
    HASH = 0x50465031
    FEATURE_NAME = INPUT_FEATURE_NAME = "Pawn_FrontierPairs"
    NUM_INPUTS = NUM_REAL_FEATURES = 7 * 49 * 49
    MAX_ACTIVE_FEATURES = 7


class PawnFilePairs(_GlobalFeature):
    HASH = 0x50464331
    FEATURE_NAME = INPUT_FEATURE_NAME = "Pawn_FilePairs"
    NUM_INPUTS = NUM_REAL_FEATURES = 8 * 60
    # Also accommodates synthetic positions with more than 16 pawns.
    MAX_ACTIVE_FEATURES = 48
