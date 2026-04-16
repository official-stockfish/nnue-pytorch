import torch
from torch import nn

from ..feature_transformer import fused_double_ft_op
from .input_feature import InputFeature


class ComposedFeatureTransformer(nn.Module):
    """Thin coordinator that wraps one or more InputFeature modules.

    Each feature owns its own weight parameters. This class owns the shared
    bias and delegates everything else to the underlying features.
    """

    def __init__(self, L1, num_psqt_buckets, feature_classes: list[type]):
        super().__init__()

        self.L1 = L1
        self.num_psqt_buckets = num_psqt_buckets
        self.num_outputs = self.L1 + self.num_psqt_buckets

        features = [fc(self.num_outputs) for fc in feature_classes]
        self.features = nn.ModuleList(features)

        self.bias = nn.Parameter(torch.empty(self.num_outputs, dtype=torch.float32))

        # Aggregate attributes from components
        self.NUM_INPUTS = sum(f.NUM_INPUTS for f in features)
        self.MAX_ACTIVE_FEATURES = sum(f.MAX_ACTIVE_FEATURES for f in features)
        self.NUM_REAL_FEATURES = sum(f.NUM_REAL_FEATURES for f in features)

        self.FEATURE_NAME = "+".join(f.FEATURE_NAME for f in features)
        self.INPUT_FEATURE_NAME = "+".join(f.INPUT_FEATURE_NAME for f in features)

        self.HASH = self._compute_hash()

        self._reset_bias()

    def _compute_hash(self) -> int:
        h = 0
        for f in self.features:
            h = ((h << 1) | (h >> 31)) & 0xFFFFFFFF
            h ^= f.HASH
        return h

    def _reset_bias(self):
        import math

        sigma = math.sqrt(1 / self.NUM_INPUTS)
        with torch.no_grad():
            self.bias.uniform_(-sigma, sigma)

    def forward(self, w_indices, w_values, b_indices, b_values, us, them, ft_max_val):
        merged_weight = torch.cat([f.merged_weight() for f in self.features], dim=0)
        return fused_double_ft_op(
            w_indices,
            w_values,
            b_indices,
            b_values,
            merged_weight,
            self.bias,
            us,
            them,
            ft_max_val,
            self.L1,
            self.num_psqt_buckets
        )

    @torch.no_grad()
    def coalesce(self) -> None:
        for f in self.features:
            f.coalesce()

    @torch.no_grad()
    def init_weights(self, nnue2score: float) -> None:
        for f in self.features:
            f.init_weights(self.num_psqt_buckets, nnue2score)

        for i in range(self.num_psqt_buckets):
            self.bias[self.L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return torch.cat([f.get_export_weights() for f in self.features], dim=0)

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        offset = 0
        for f in self.features:
            n = f.NUM_REAL_FEATURES
            f.load_export_weights(export_weight[offset : offset + n])
            offset += n

    def clip_weights(self, quantization) -> None:
        for f in self.features:
            f.clip_weights(quantization)


def combine_input_features(*feature_classes: type):
    """Return a factory that creates a ComposedFeatureTransformer."""

    def factory(L1, num_psqt_buckets) -> ComposedFeatureTransformer:
        return ComposedFeatureTransformer(L1, num_psqt_buckets, feature_classes)

    return factory
