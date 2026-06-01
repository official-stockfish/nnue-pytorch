import torch
from torch import nn

from typing import Callable

from ..feature_transformer import SparseLinearFunction
from .input_feature import InputFeature

from ...quantize import QuantizationManager

class ComposedFeatureTransformer(nn.Module):
    """Thin coordinator that wraps one or more InputFeature modules.

    Each feature owns its own weight parameters. This class owns the shared
    bias and delegates everything else to the underlying features.
    """

    def __init__(self, feature_classes: list[Callable[[int], InputFeature]], l1_size: int, num_psqt_buckets:int, quantization: QuantizationManager):
        super().__init__()

        self.l1_size = l1_size
        self.num_psqt_buckets = num_psqt_buckets
        self.num_outputs = l1_size + num_psqt_buckets

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

        self.quantization = quantization

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

    def forward(
        self,
        feature_indices_0,
        feature_values_0,
        feature_indices_1,
        feature_values_1,
        fake_quantize_weights: bool=False,
    ):
        merged = torch.cat([f.merged_weight() for f in self.features], dim=0)
        bias = self.bias
        if fake_quantize_weights:
            w  = self.quantization.fake_quantize_weights(merged[:, :self.l1_size], "ft_weight")
            pw = self.quantization.fake_quantize_weights(merged[:, self.l1_size:], "ft_psqt_weight")
            merged = torch.cat([w, pw], dim=1)
            # Technically unnecessary to zero bias, but it makes it clearer that the PSQT part of the bias is not used.
            bias[self.l1_size:].zero_()
            bias = self.quantization.fake_quantize_weights(bias, "ft_bias")
        return (
            SparseLinearFunction.apply(
                feature_indices_0,
                feature_values_0,
                merged,
                bias,
            ),
            SparseLinearFunction.apply(
                feature_indices_1,
                feature_values_1,
                merged,
                bias,
            ),
        )

    @torch.no_grad()
    def coalesce(self) -> None:
        for f in self.features:
            f.coalesce()

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        for f in self.features:
            f.zero_virtual_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        num_psqt_buckets = self.num_psqt_buckets
        for f in self.features:
            f.init_weights(num_psqt_buckets, self.quantization.nnue2score)

        L1 = self.num_outputs - num_psqt_buckets
        for i in range(num_psqt_buckets):
            self.bias[L1 + i] = 0.0

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
