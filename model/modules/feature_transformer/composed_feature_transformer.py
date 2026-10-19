from collections.abc import Callable

import torch
from torch import nn

from ...quantize import QuantizationManager
from ..features.input_feature import InputFeature
from .sparse_linear_functions import SparseLinearFunction


class ComposedFeatureTransformer(nn.Module):
    """Thin coordinator that wraps one or more InputFeature modules.

    Each feature owns its own weight parameters. This class owns the shared
    bias and delegates everything else to the underlying features.
    """

    def __init__(
        self,
        feature_classes: list[Callable[[int], InputFeature]],
        l1_size: int,
        quantization: QuantizationManager,
        l1_bias: int = 256,
        l2_size: int = 32,
        l3_size: int = 32,
    ):
        super().__init__()

        if not l1_size % 2 == 0:
            raise ValueError(f"l1_size must be even, got {l1_size}.")

        self.l1_size = l1_size
        self.l1_bias = l1_bias
        self.l2_size = l2_size
        self.l3_size = l3_size
        self.num_outputs = l1_size + l1_bias

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

    def merged_weight_and_bias(
        self,
        fake_quantize_weights: bool = False,
    ):
        merged = torch.cat([f.merged_weight() for f in self.features], dim=0)
        b = self.bias
        if fake_quantize_weights:
            merged = self.quantization.fake_quantize_weights(merged, "ft_weight")
            b = self.quantization.fake_quantize_weights(b, "ft_bias")

        return merged, b

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
        for f in self.features:
            f.init_weights()

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

    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
        backend: str = "auto",
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        merged, bias = self.merged_weight_and_bias(fake_quantize_weights)
        ft_max_act = self.quantization.max_ft_activation

        w = SparseLinearFunction.apply(white_indices, merged, bias, backend=backend)
        b = SparseLinearFunction.apply(black_indices, merged, bias, backend=backend)

        w_main = w[:, : self.l1_size]
        b_main = b[:, : self.l1_size]
        l0_ = (us * torch.cat([w_main, b_main], dim=1)) + (
            them * torch.cat([b_main, w_main], dim=1)
        )
        l0_ = torch.clamp(l0_, 0.0, ft_max_act)
        l0_s = torch.split(l0_, self.l1_size // 2, dim=1)
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
        l0_ = torch.cat(l0_s1, dim=1)

        if fake_quantize_acts:
            l0_ = self.quantization.fake_quantize_ft_act(l0_)
        l0_ = l0_ * self.quantization.l0_correction_factor

        w_dyn = w[:, self.l1_size :]
        b_dyn = b[:, self.l1_size :]
        a_stm = (us - them) * (w_dyn - b_dyn)
        half_dyn = self.l1_bias // 2
        u = a_stm[:, :half_dyn]
        v = a_stm[:, half_dyn:]

        if fake_quantize_acts:
            dyn_biases = self.quantization.fake_quantize_dyn_biases(
                u, v, self.l2_size, self.l3_size
            )
        else:
            gate = torch.clamp(u, 0.0, ft_max_act)
            delta_b = gate * v
            l2 = self.l2_size
            l3 = self.l3_size
            dyn_biases = (
                delta_b[:, :l2],
                delta_b[:, l2 : 2 * l2],
                delta_b[:, 2 * l2 : 2 * l2 + l3],
                delta_b[:, 2 * l2 + l3 : 2 * (l2 + l3)],
            )

        return l0_, dyn_biases
