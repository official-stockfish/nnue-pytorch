import torch
from torch import nn
import torch.nn.functional as F


class StackedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, count: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.count = count
        self.linear = nn.Linear(in_features, out_features * count)

        self._init_uniformly()

    @torch.no_grad()
    def _init_uniformly(self) -> None:
        init_weight = self.linear.weight[0 : self.out_features, :]
        init_bias = self.linear.bias[0 : self.out_features]

        self.linear.weight.copy_(init_weight.repeat(self.count, 1))
        self.linear.bias.copy_(init_bias.repeat(self.count))

    def forward(self, x: torch.Tensor, ls_indices: torch.Tensor) -> torch.Tensor:
        stacked_output = self.linear(x)

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
    def __init__(self, in_features: int, out_features: int, count: int):
        super().__init__(in_features, out_features, count)

        self.factorized_linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            self.factorized_linear.weight.zero_()
            self.factorized_linear.bias.zero_()

    def forward(self, x: torch.Tensor, ls_indices: torch.Tensor) -> torch.Tensor:
        merged_weight = self.linear.weight + self.factorized_linear.weight.repeat(
            self.count, 1
        )
        merged_bias = self.linear.bias + self.factorized_linear.bias.repeat(self.count)

        stacked_output = F.linear(x, merged_weight, merged_bias)

        return self.select_output(stacked_output, ls_indices)

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

        self.factorized_linear.weight.zero_()
        self.factorized_linear.bias.zero_()
