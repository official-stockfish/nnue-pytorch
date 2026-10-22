import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..quantize import QuantizationManager
from .router import CODEBOOK_DIM

try:
    from .grouped_linear import grouped_l1, grouped_l1_preferred
except (ImportError, OSError, RuntimeError):
    grouped_l1 = None


class StackedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        count: int,
        quantization: QuantizationManager | None = None,
        layer_key: str | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.count = count
        self.linear = nn.Linear(in_features, out_features * count)

        self.quantization = quantization
        self.layer_key = layer_key

        self._init_uniformly()

    @torch.no_grad()
    def _init_uniformly(self) -> None:
        init_weight = self.linear.weight[0 : self.out_features, :]
        init_bias = self.linear.bias[0 : self.out_features]

        self.linear.weight.copy_(init_weight.repeat(self.count, 1))
        self.linear.bias.copy_(init_bias.repeat(self.count))

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_weights: bool = False,
    ) -> torch.Tensor:
        weight = self.linear.weight
        bias = self.linear.bias
        if fake_quantize_weights:
            if self.quantization is None or self.layer_key is None:
                raise RuntimeError(
                    "self.quantization and self.layer_key are required to use fake quantize weights."
                )
            weight = self.quantization.fake_quantize_weights(
                weight, f"{self.layer_key}_weight"
            )
            bias = self.quantization.fake_quantize_weights(
                bias, f"{self.layer_key}_bias"
            )

        stacked_output = F.linear(x, weight, bias)

        return self.select_output(stacked_output, ls_indices)

    def select_output(
        self, stacked_output: torch.Tensor, ls_indices: torch.Tensor
    ) -> torch.Tensor:
        reshaped_output = stacked_output.reshape(-1, self.out_features)

        idx_offset = torch.arange(
            0,
            ls_indices.shape[0] * self.count,
            self.count,
            device=stacked_output.device,
        )
        indices = ls_indices.flatten() + idx_offset

        selected_output = reshaped_output[indices]

        return selected_output

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        layer = nn.Linear(self.in_features, self.out_features)

        begin = index * self.out_features
        end = (index + 1) * self.out_features

        layer.weight.copy_(self.linear.weight[begin:end, :])
        layer.bias.copy_(self.linear.bias[begin:end])

        return layer


class FactorizedStackedLinear(StackedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        count: int,
        quantization: QuantizationManager,
        layer_key: str,
    ):
        super().__init__(in_features, out_features, count, quantization, layer_key)

        self.factorized_linear = nn.Linear(in_features, out_features)
        self.zero_virtual_weights()

    def forward(
        self,
        x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_weights: bool = False,
    ) -> torch.Tensor:
        merged_weight = self.linear.weight + self.factorized_linear.weight.repeat(
            self.count, 1
        )
        merged_bias = self.linear.bias + self.factorized_linear.bias.repeat(self.count)

        if fake_quantize_weights:
            if self.quantization is None or self.layer_key is None:
                raise RuntimeError(
                    "self.quantization and self.layer_key are required to use fake quantize weights."
                )
            merged_weight = self.quantization.fake_quantize_weights(
                merged_weight, f"{self.layer_key}_weight"
            )
            merged_bias = self.quantization.fake_quantize_weights(
                merged_bias, f"{self.layer_key}_bias"
            )

        # The custom kernels specialize FP32 K->N, N <= 128. The router uses
        # one of its 256 threads per bucket for initialization/reservation.
        # They only outperform the dense path with enough stacks (measured
        # crossover: 32; NNUE_GROUPED_L1=1|0 overrides). Keep the standard
        # path for missing dependencies and other workloads.
        if (
            grouped_l1 is not None
            and x.is_cuda
            and torch.version.hip is None
            and x.dtype == merged_weight.dtype == merged_bias.dtype == torch.float32
            and not torch.is_autocast_enabled()
            and 128 <= self.in_features <= 4096
            and self.in_features % 128 == 0
            and 1 <= self.out_features <= 128
            and 1 <= self.count <= 256
            and grouped_l1_preferred(self.count)
            and x.ndim == 2
            and x.shape[0] > 0
            and x.shape[1] == self.in_features
            and torch.cuda.get_device_capability(x.device)[0] >= 8
        ):
            return grouped_l1(x, merged_weight, merged_bias, ls_indices, self.count)

        stacked_output = F.linear(x, merged_weight, merged_bias)

        return self.select_output(stacked_output, ls_indices)

    @torch.no_grad()
    def zero_virtual_weights(self):
        self.factorized_linear.weight.zero_()
        self.factorized_linear.bias.zero_()

    @torch.no_grad()
    def at_index(self, index: int) -> nn.Linear:
        layer = super().at_index(index)

        layer.weight.add_(self.factorized_linear.weight)
        layer.bias.add_(self.factorized_linear.bias)

        return layer

    @torch.no_grad()
    def coalesce_weights(self) -> None:
        for i in range(self.count):
            begin = i * self.out_features
            end = (i + 1) * self.out_features

            self.linear.weight[begin:end, :].add_(self.factorized_linear.weight)
            self.linear.bias[begin:end].add_(self.factorized_linear.bias)

        self.zero_virtual_weights()


class LowRankStackedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        count: int,
        rank: int = 32,
        codebook_dim: int = CODEBOOK_DIM,
        quantization: QuantizationManager | None = None,
        layer_key: str | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.count = count
        self.rank = rank
        self.codebook_dim = codebook_dim
        self.quantization = quantization
        self.layer_key = layer_key
        self.is_coalesced = False

        self.adapter_down = nn.Parameter(torch.empty(rank, codebook_dim))
        self.adapter_up = nn.Parameter(torch.empty(out_features * in_features, rank))
        self.bias_adapter = nn.Parameter(torch.empty(out_features, rank))

        self.register_buffer(
            "coalesced_weight",
            torch.zeros(count * out_features, in_features, dtype=torch.float32),
        )
        self.register_buffer(
            "coalesced_bias",
            torch.zeros(count * out_features, dtype=torch.float32),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound_cb = 1.0 / math.sqrt(self.codebook_dim) if self.codebook_dim > 0 else 0.0
        nn.init.uniform_(self.adapter_down, -bound_cb, bound_cb)

        bound_up = (
            1.0 / math.sqrt(self.in_features * self.rank)
            if self.in_features > 0 and self.rank > 0
            else 0.0
        )
        nn.init.uniform_(self.adapter_up, -bound_up, bound_up)

        bound_bias = 1.0 / math.sqrt(self.rank) if self.rank > 0 else 0.0
        nn.init.uniform_(self.bias_adapter, -bound_bias, bound_bias)

    def get_delta_weights(self, codebook: torch.Tensor) -> torch.Tensor:
        # codebook: (count, codebook_dim)
        if codebook.shape[-1] != self.adapter_down.shape[1]:
            raise ValueError(
                f"codebook dimension mismatch in LowRankStackedLinear: "
                f"adapter_down expects {self.adapter_down.shape[1]}, got {codebook.shape[-1]}"
            )
        act = F.gelu(codebook)
        h = F.linear(act, self.adapter_down)  # (count, rank)
        w = F.linear(h, self.adapter_up)  # (count, out_features * in_features)
        return w.view(-1, self.out_features, self.in_features)

    def get_delta_biases(self, codebook: torch.Tensor) -> torch.Tensor:
        # codebook: (count, codebook_dim)
        if codebook.shape[-1] != self.adapter_down.shape[1]:
            raise ValueError(
                f"codebook dimension mismatch in LowRankStackedLinear: "
                f"adapter_down expects {self.adapter_down.shape[1]}, got {codebook.shape[-1]}"
            )
        act = F.gelu(codebook)
        h = F.linear(act, self.adapter_down)  # (count, rank)
        return F.linear(h, self.bias_adapter)  # (count, out_features)

    def get_merged_weights_and_biases(
        self, codebook: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_coalesced:
            return self.coalesced_weight, self.coalesced_bias

        if codebook is None:
            codebook = torch.zeros(
                self.count,
                self.codebook_dim,
                device=self.adapter_down.device,
                dtype=self.adapter_down.dtype,
            )

        w = self.get_delta_weights(codebook)
        b = self.get_delta_biases(codebook)

        flat_w = w.reshape(
            self.count * self.out_features, self.in_features
        ).contiguous()
        flat_b = b.reshape(self.count * self.out_features).contiguous()
        return flat_w, flat_b

    def forward(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor | None,
        ls_indices: torch.Tensor,
        fake_quantize_weights: bool = False,
    ) -> torch.Tensor:
        flat_w, flat_b = self.get_merged_weights_and_biases(codebook)

        if fake_quantize_weights:
            if self.quantization is None or self.layer_key is None:
                raise RuntimeError(
                    "self.quantization and self.layer_key are required to use fake quantize weights."
                )
            flat_w = self.quantization.fake_quantize_weights(
                flat_w, f"{self.layer_key}_weight"
            )
            flat_b = self.quantization.fake_quantize_weights(
                flat_b, f"{self.layer_key}_bias"
            )

        if (
            grouped_l1 is not None
            and x.is_cuda
            and torch.version.hip is None
            and x.dtype == flat_w.dtype == flat_b.dtype == torch.float32
            and not torch.is_autocast_enabled()
            and 128 <= self.in_features <= 4096
            and self.in_features % 128 == 0
            and 1 <= self.out_features <= 128
            and 1 <= self.count <= 256
            and grouped_l1_preferred(self.count)
            and x.ndim == 2
            and x.shape[0] > 0
            and x.shape[1] == self.in_features
            and torch.cuda.get_device_capability(x.device)[0] >= 8
        ):
            return grouped_l1(x, flat_w, flat_b, ls_indices, self.count)

        stacked_output = F.linear(x, flat_w, flat_b)
        return self.select_output(stacked_output, ls_indices)

    def select_output(
        self, stacked_output: torch.Tensor, ls_indices: torch.Tensor
    ) -> torch.Tensor:
        reshaped_output = stacked_output.reshape(-1, self.out_features)

        idx_offset = torch.arange(
            0,
            ls_indices.shape[0] * self.count,
            self.count,
            device=stacked_output.device,
        )
        indices = ls_indices.flatten() + idx_offset

        selected_output = reshaped_output[indices]
        return selected_output

    @torch.no_grad()
    def at_index(self, index: int, codebook: torch.Tensor | None = None) -> nn.Linear:
        layer = nn.Linear(self.in_features, self.out_features)

        if self.is_coalesced:
            flat_w = self.coalesced_weight
            flat_b = self.coalesced_bias
        else:
            flat_w, flat_b = self.get_merged_weights_and_biases(codebook)
            if self.quantization is not None and self.layer_key is not None:
                scale_key = f"{self.layer_key}_weight"
                if scale_key in self.quantization.weight_scales_dict:
                    max_w = 127.0 / self.quantization.weight_scales_dict[scale_key]
                    flat_w = flat_w.clamp(-max_w, max_w)

        begin = index * self.out_features
        end = (index + 1) * self.out_features

        layer.weight.copy_(flat_w[begin:end, :])
        layer.bias.copy_(flat_b[begin:end])
        return layer

    @torch.no_grad()
    def coalesce_weights(self, codebook: torch.Tensor | None = None) -> None:
        flat_w, flat_b = self.get_merged_weights_and_biases(codebook)
        if self.quantization is not None and self.layer_key is not None:
            scale_key = f"{self.layer_key}_weight"
            if scale_key in self.quantization.weight_scales_dict:
                max_w = 127.0 / self.quantization.weight_scales_dict[scale_key]
                flat_w = flat_w.clamp(-max_w, max_w)
        self.coalesced_weight.copy_(flat_w)
        self.coalesced_bias.copy_(flat_b)
        self.is_coalesced = True

    @torch.no_grad()
    def init_coalesced_storage(self) -> None:
        self.coalesced_weight.zero_()
        self.coalesced_bias.zero_()
        self.adapter_down.zero_()
        self.adapter_up.zero_()
        self.bias_adapter.zero_()
        self.is_coalesced = True

    @torch.no_grad()
    def permute_inputs(self, permutation: Sequence[int]) -> None:
        reshaped = self.adapter_up.view(self.out_features, self.in_features, self.rank)
        permuted = reshaped[:, permutation, :].contiguous()
        self.adapter_up.copy_(
            permuted.view(self.out_features * self.in_features, self.rank)
        )
        if self.is_coalesced:
            self.coalesced_weight.copy_(self.coalesced_weight[:, permutation])
