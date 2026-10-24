import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..quantize import QuantizationManager
from .router import CODEBOOK_DIM, ieee_matmul_precision, rms_norm

try:
    from .grouped_linear import grouped_l1, grouped_l1_preferred
except (ImportError, OSError, RuntimeError):
    grouped_l1 = None


class MoeLinear(nn.Module):
    """
    Mixture-of-Experts (MoE) Linear layer with low-rank adapter projection from codebook embeddings.
    Dynamically generates expert weights and biases on the fly or accesses coalesced tables.
    """

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

    def get_expert_weight_and_bias(
        self, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps a 1D context vector c (codebook_dim,) to single expert (weight, bias).
        """
        if c.shape[-1] != self.adapter_down.shape[1]:
            raise ValueError(
                f"codebook dimension mismatch in MoeLinear: "
                f"adapter_down expects {self.adapter_down.shape[1]}, got {c.shape[-1]}"
            )
        with ieee_matmul_precision():
            act = F.gelu(c)
            h = F.linear(act, self.adapter_down)
            h = rms_norm(h, target_norm=1.0)
            w = F.linear(h, self.adapter_up).reshape(
                self.out_features, self.in_features
            )
            b = F.linear(h, self.bias_adapter).reshape(self.out_features)
            return w, b

    def get_delta_weights(self, codebook: torch.Tensor) -> torch.Tensor:
        # codebook: (count, codebook_dim)
        if codebook.shape[-1] != self.adapter_down.shape[1]:
            raise ValueError(
                f"codebook dimension mismatch in MoeLinear: "
                f"adapter_down expects {self.adapter_down.shape[1]}, got {codebook.shape[-1]}"
            )
        act = F.gelu(codebook)
        h = F.linear(act, self.adapter_down)  # (count, rank)
        h = rms_norm(h, target_norm=1.0)
        w = F.linear(h, self.adapter_up)  # (count, out_features * in_features)
        return w.view(-1, self.out_features, self.in_features)

    def get_delta_biases(self, codebook: torch.Tensor) -> torch.Tensor:
        # codebook: (count, codebook_dim)
        if codebook.shape[-1] != self.adapter_down.shape[1]:
            raise ValueError(
                f"codebook dimension mismatch in MoeLinear: "
                f"adapter_down expects {self.adapter_down.shape[1]}, got {codebook.shape[-1]}"
            )
        act = F.gelu(codebook)
        h = F.linear(act, self.adapter_down)  # (count, rank)
        h = rms_norm(h, target_norm=1.0)
        return F.linear(h, self.bias_adapter)  # (count, out_features)

    def get_all_weights_and_biases(
        self, codebook: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_coalesced:
            return self.coalesced_weight, self.coalesced_bias

        if codebook is None:
            raise ValueError(
                "codebook must be provided when MoeLinear is not coalesced"
            )

        with ieee_matmul_precision():
            w = self.get_delta_weights(codebook)
            b = self.get_delta_biases(codebook)

            flat_w = w.reshape(
                self.count * self.out_features, self.in_features
            ).contiguous()
            flat_b = b.reshape(self.count * self.out_features).contiguous()
            return flat_w, flat_b

    # Backward compatibility alias
    get_merged_weights_and_biases = get_all_weights_and_biases

    def forward(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor | None,
        expert_indices: torch.Tensor,
        fake_quantize_weights: bool = False,
    ) -> torch.Tensor:
        flat_w, flat_b = self.get_all_weights_and_biases(codebook)

        if fake_quantize_weights:
            if self.quantization is None or self.layer_key is None:
                raise RuntimeError(
                    "self.quantization and self.layer_key are required to use fake quantize weights."
                )
            flat_w = self.quantization.clip_ls_weight(flat_w, self.layer_key)
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
            return grouped_l1(x, flat_w, flat_b, expert_indices, self.count)

        stacked_output = F.linear(x, flat_w, flat_b)
        return self.select_output(stacked_output, expert_indices)

    def select_output(
        self, stacked_output: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        reshaped_output = stacked_output.reshape(-1, self.out_features)

        idx_offset = torch.arange(
            0,
            expert_indices.shape[0] * self.count,
            self.count,
            device=stacked_output.device,
        )
        indices = expert_indices.flatten() + idx_offset

        selected_output = reshaped_output[indices]
        return selected_output

    @torch.no_grad()
    def at_index(self, index: int, codebook: torch.Tensor | None = None) -> nn.Linear:
        layer = nn.Linear(self.in_features, self.out_features)

        if self.is_coalesced:
            flat_w = self.coalesced_weight
            flat_b = self.coalesced_bias
        else:
            if codebook is None:
                raise ValueError(
                    "codebook must be provided when MoeLinear is not coalesced"
                )
            flat_w, flat_b = self.get_all_weights_and_biases(codebook)
            if self.quantization is not None and self.layer_key is not None:
                flat_w = self.quantization.clip_ls_weight(flat_w, self.layer_key)

        begin = index * self.out_features
        end = (index + 1) * self.out_features

        layer.weight.copy_(flat_w[begin:end, :])
        layer.bias.copy_(flat_b[begin:end])
        return layer

    @torch.no_grad()
    def coalesce_weights(self, codebook: torch.Tensor) -> None:
        flat_w, flat_b = self.get_all_weights_and_biases(codebook)
        if self.quantization is not None and self.layer_key is not None:
            flat_w = self.quantization.clip_ls_weight(flat_w, self.layer_key)
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


# Backward compatibility aliases
LowRankStackedLinear = MoeLinear
