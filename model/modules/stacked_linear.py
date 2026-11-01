import torch
import torch.nn.functional as F
from torch import nn

from ..quantize import QuantizationManager

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

        self.group_factor_w = nn.ParameterList()
        self.group_factor_b = nn.ParameterList()
        self.group_maps: list[torch.Tensor] = []

        self._init_uniformly()

    def set_group_factors(self, maps: list[torch.Tensor]) -> None:
        """maps[f][stack] = group id of the stack under factor f (zero-initialised tables)."""
        for m in maps:
            g = int(m.max()) + 1
            self.group_factor_w.append(
                nn.Parameter(torch.zeros(g * self.out_features, self.in_features))
            )
            self.group_factor_b.append(nn.Parameter(torch.zeros(g * self.out_features)))
            self.register_buffer(
                f"group_map_{len(self.group_maps)}", m.clone().long(), persistent=False
            )
            self.group_maps.append(m.clone().long())

    @torch.no_grad()
    def coalesce_group_factors(self) -> None:
        ge = self.group_expansion()
        if ge is None:
            return
        self.linear.weight.add_(ge[0])
        self.linear.bias.add_(ge[1])
        for fw, fb in zip(self.group_factor_w, self.group_factor_b):
            fw.zero_()
            fb.zero_()

    def group_expansion(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(count*out, in) weight and (count*out,) bias contributions of the group factors, or None."""
        if len(self.group_maps) == 0:
            return None
        w = 0
        b = 0
        for f, (fw, fb) in enumerate(zip(self.group_factor_w, self.group_factor_b)):
            m = getattr(self, f"group_map_{f}").to(fw.device)
            w = w + fw.view(-1, self.out_features, self.in_features)[m].reshape(
                self.count * self.out_features, self.in_features
            )
            b = b + fb.view(-1, self.out_features)[m].reshape(
                self.count * self.out_features
            )
        return w, b

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
        ge = self.group_expansion()
        if ge is not None:
            weight = weight + ge[0]
            bias = bias + ge[1]
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
        ge = self.group_expansion()
        if ge is not None:
            layer.weight.add_(ge[0][begin:end, :])
            layer.bias.add_(ge[1][begin:end])

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
        ge = self.group_expansion()
        if ge is not None:
            merged_weight = merged_weight + ge[0]
            merged_bias = merged_bias + ge[1]

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
        ge = self.group_expansion()
        if ge is not None:
            self.linear.weight.add_(ge[0])
            self.linear.bias.add_(ge[1])
            for fw, fb in zip(self.group_factor_w, self.group_factor_b):
                fw.zero_()
                fb.zero_()

        self.zero_virtual_weights()
