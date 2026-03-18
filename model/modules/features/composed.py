import torch
from torch import nn

from ..feature_transformer import SparseLinearFunction
from .input_feature import InputFeature


class ComposedFeatureTransformer(nn.Module):
    """Thin coordinator that wraps one or more InputFeature modules.

    Each feature owns its own weight parameters. This class owns the shared
    bias and delegates everything else to the underlying features.
    """

    def __init__(self, features: list[InputFeature]):
        super().__init__()

        self.features = nn.ModuleList(features)

        self.l1 = features[0].l1
        self.num_psqt_buckets = features[0].num_psqt_buckets
        self.num_outputs = self.l1 + self.num_psqt_buckets

        # Split the shared bias into FT and PSQT parameter groups
        self.bias_ft = nn.Parameter(torch.empty(self.l1, dtype=torch.float32))
        self.bias_psqt = nn.Parameter(torch.empty(self.num_psqt_buckets, dtype=torch.float32))

        # Aggregate attributes from components
        self.NUM_INPUTS = sum(f.NUM_INPUTS for f in features)
        self.MAX_ACTIVE_FEATURES = sum(f.MAX_ACTIVE_FEATURES for f in features)
        self.NUM_REAL_FEATURES = sum(f.NUM_REAL_FEATURES for f in features)

        self.FEATURE_NAME = "+".join(f.FEATURE_NAME for f in features)
        self.INPUT_FEATURE_NAME = "+".join(f.INPUT_FEATURE_NAME for f in features)

        self.HASH = self._compute_hash()

        self._reset_bias()

    def get_ft_params(self, include_bias=True, bias_only=False) -> list[nn.Parameter]:
        """Aggregate FT parameters from all features and the shared FT bias."""
        params = [self.bias_ft] if include_bias else []
        if not bias_only:
            for f in self.features:
                params.extend(f.get_ft_params())
        return params

    def get_pqst_params(self, include_bias=True, bias_only=False) -> list[nn.Parameter]:
        """Aggregate PSQT parameters from all features and the shared PSQT bias."""
        params = [self.bias_psqt] if include_bias else []
        if not bias_only:
            for f in self.features:
                params.extend(f.get_pqst_params())
        return params

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
            self.bias_ft.uniform_(-sigma, sigma)
            self.bias_psqt.uniform_(-sigma, sigma)

    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        merged_weights = torch.cat([f.merged_weight() for f in self.features], dim=0)
        merged_bias = torch.cat([self.bias_ft, self.bias_psqt], dim=0)

        return (
            SparseLinearFunction.apply(
                feature_indices_0,
                feature_values_0,
                merged_weights,
                merged_bias,
            ),
            SparseLinearFunction.apply(
                feature_indices_1,
                feature_values_1,
                merged_weights,
                merged_bias,
            ),
        )

    @torch.no_grad()
    def coalesce(self) -> None:
        for f in self.features:
            f.coalesce()

    @torch.no_grad()
    def init_weights(self, nnue2score: float) -> None:
        for f in self.features:
            f.init_weights(nnue2score)

        # PSQT bias starts at 0
        self.bias_psqt.zero_()

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

    def factory(l1: int, num_psqt_buckets: int) -> ComposedFeatureTransformer:
        features = [fc(l1, num_psqt_buckets) for fc in feature_classes]
        return ComposedFeatureTransformer(features)

    return factory
