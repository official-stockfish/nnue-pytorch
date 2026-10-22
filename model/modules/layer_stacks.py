from collections.abc import Generator

import torch
from torch import nn

from ..quantize import QuantizationManager
from .config import LayerStacksConfig
from .moe_linear import MoeLinear
from .router import CODEBOOK_DIM


class LayerStacks(nn.Module):
    def __init__(
        self,
        count: int,
        config: LayerStacksConfig,
        quantization: QuantizationManager,
        rank: int = 32,
        codebook_dim: int = CODEBOOK_DIM,
    ):
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3
        self.quantization = quantization
        self.rank = rank
        self.codebook_dim = codebook_dim

        self.l1 = MoeLinear(
            2 * self.L1 // 2,
            self.L2,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_l1",
        )
        self.l2 = MoeLinear(
            self.L2 * 2,
            self.L3,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_l2",
        )
        self.output = MoeLinear(
            self.L2 * 2 + self.L3 * 2,
            1,
            count,
            rank=rank,
            codebook_dim=codebook_dim,
            quantization=quantization,
            layer_key="ls_output",
        )

        with torch.no_grad():
            self.output.bias_adapter.zero_()

    def get_expert_deltas(self, codebook: torch.Tensor) -> list[torch.Tensor]:
        return [
            self.l1.get_delta_weights(codebook),
            self.l2.get_delta_weights(codebook),
            self.output.get_delta_weights(codebook),
        ]

    def forward_head(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor | None,
        expert_indices: torch.Tensor,
        dyn_biases: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        b1_sqr, b1_lin, b2_sqr, b2_lin = dyn_biases
        l1c_ = self.l1(x, codebook, expert_indices, fake_quantize_weights)

        # Extract the short-path skip connection before fake quantization and dynamic biases
        l1x_out = l1c_[:, -2].view(-1, 1) - l1c_[:, -1].view(-1, 1)

        l1x_sqr = l1c_ + b1_sqr
        l1x_lin = l1c_ + b1_lin

        l1_sqr = torch.pow(l1x_sqr, 2.0)
        if fake_quantize_acts:
            l1_sqr = self.quantization.fake_quantize_ls_act(l1_sqr)

        l1_sqr = l1_sqr * (self.quantization.sqr_crelu_correction_factor)

        l1x_ = l1x_lin
        if fake_quantize_acts:
            l1x_ = self.quantization.fake_quantize_ls_act(l1x_)

        l1x_ = torch.cat([l1_sqr, l1x_], dim=1)
        l1x_ = self.quantization.clip_ls_act(l1x_)

        l2c_ = self.l2(l1x_, codebook, expert_indices, fake_quantize_weights)

        l2x_sqr = l2c_ + b2_sqr
        l2x_lin = l2c_ + b2_lin

        l2_sqr = torch.pow(l2x_sqr, 2.0)
        if fake_quantize_acts:
            l2_sqr = self.quantization.fake_quantize_ls_act(l2_sqr)

        l2_sqr = l2_sqr * (self.quantization.sqr_crelu_correction_factor)

        l2x_ = l2x_lin
        if fake_quantize_acts:
            l2x_ = self.quantization.fake_quantize_ls_act(l2x_)

        l2x_ = torch.cat([l2_sqr, l2x_], dim=1)
        l2x_ = self.quantization.clip_ls_act(l2x_)

        l3_input = torch.cat([l1x_, l2x_], dim=1)

        l3c_ = self.output(l3_input, codebook, expert_indices, fake_quantize_weights)

        if fake_quantize_acts:
            l1x_out = self.quantization.fake_quantize_skip_act(l1x_out)

        l3x_ = l3c_ + l1x_out
        if fake_quantize_acts:
            l3x_ = self.quantization.fake_quantize_output(l3x_)

        assert l3x_.shape[1] == 1, (
            f"Expected output shape (batch_size, 1), got {l3x_.shape}"
        )
        return l3x_

    def forward(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor | None,
        e1: torch.Tensor,
        e2: torch.Tensor,
        dyn_biases_1: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dyn_biases_2: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        fake_quantize_acts: bool = True,
        fake_quantize_weights: bool = True,
    ) -> torch.Tensor:
        out1 = self.forward_head(
            x,
            codebook,
            e1,
            dyn_biases_1,
            fake_quantize_acts=fake_quantize_acts,
            fake_quantize_weights=fake_quantize_weights,
        )
        out2 = self.forward_head(
            x,
            codebook,
            e2,
            dyn_biases_2,
            fake_quantize_acts=fake_quantize_acts,
            fake_quantize_weights=fake_quantize_weights,
        )
        return out1 + out2

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
        codebook: torch.Tensor | None = None,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        if not self.l1.is_coalesced:
            if codebook is None:
                raise ValueError(
                    "codebook must be provided when layer stacks are not coalesced"
                )
            w1, b1 = self.l1.get_all_weights_and_biases(codebook)
            w2, b2 = self.l2.get_all_weights_and_biases(codebook)
            wo, bo = self.output.get_all_weights_and_biases(codebook)
            if self.quantization is not None:
                w1 = self.quantization.clip_ls_weight(w1, self.l1.layer_key)
                w2 = self.quantization.clip_ls_weight(w2, self.l2.layer_key)
                wo = self.quantization.clip_ls_weight(wo, self.output.layer_key)
        else:
            w1, b1 = self.l1.coalesced_weight, self.l1.coalesced_bias
            w2, b2 = self.l2.coalesced_weight, self.l2.coalesced_bias
            wo, bo = self.output.coalesced_weight, self.output.coalesced_bias

        for i in range(self.count):
            l1 = nn.Linear(self.l1.in_features, self.l1.out_features)
            l2 = nn.Linear(self.l2.in_features, self.l2.out_features)
            output = nn.Linear(self.output.in_features, self.output.out_features)

            l1.weight.copy_(
                w1[i * self.l1.out_features : (i + 1) * self.l1.out_features, :]
            )
            l1.bias.copy_(b1[i * self.l1.out_features : (i + 1) * self.l1.out_features])

            l2.weight.copy_(
                w2[i * self.l2.out_features : (i + 1) * self.l2.out_features, :]
            )
            l2.bias.copy_(b2[i * self.l2.out_features : (i + 1) * self.l2.out_features])

            output.weight.copy_(
                wo[i * self.output.out_features : (i + 1) * self.output.out_features, :]
            )
            output.bias.copy_(
                bo[i * self.output.out_features : (i + 1) * self.output.out_features]
            )

            yield (l1, l2, output)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self, codebook: torch.Tensor) -> None:
        self.l1.coalesce_weights(codebook)
        self.l2.coalesce_weights(codebook)
        self.output.coalesce_weights(codebook)
