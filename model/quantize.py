from dataclasses import dataclass
from typing import Callable, NotRequired, TypedDict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import NNUEModel


class WeightClippingConfig(TypedDict):
    params: list[torch.Tensor]
    min_weight: float
    max_weight: float
    virtual_params: NotRequired[torch.Tensor]


@dataclass
class QuantizationConfig:
    nnue2score: float = 600.0
    weight_scale_hidden: float = 64.0
    weight_scale_out: float = 16.0
    quantized_one: float = 127.0


class QuantizationManager:
    def __init__(self, config: QuantizationConfig):
        self.nnue2score = config.nnue2score
        self.weight_scale_hidden = config.weight_scale_hidden
        self.weight_scale_out = config.weight_scale_out
        self.quantized_one = config.quantized_one

        self.max_hidden_weight = self.quantized_one / self.weight_scale_hidden
        self.max_out_weight = (self.quantized_one * self.quantized_one) / (
            self.nnue2score * self.weight_scale_out
        )

    def generate_weight_clipping_config(
        self, model: "NNUEModel"
    ) -> list[WeightClippingConfig]:
        return [
            {
                "params": [model.layer_stacks.l1.linear.weight],
                "min_weight": -self.max_hidden_weight,
                "max_weight": self.max_hidden_weight,
                "virtual_params": model.layer_stacks.l1.factorized_linear.weight,
            },
            {
                "params": [model.layer_stacks.l2.linear.weight],
                "min_weight": -self.max_hidden_weight,
                "max_weight": self.max_hidden_weight,
            },
            {
                "params": [model.layer_stacks.output.linear.weight],
                "min_weight": -self.max_out_weight,
                "max_weight": self.max_out_weight,
            },
        ]

    def quantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
        callback: Callable = lambda *args, **kwargs: None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = bias.mul(self.quantized_one).round().to(torch.int16)
        weight = weight.mul(self.quantized_one).round().to(torch.int16)
        psqt_weight = (
            psqt_weight.mul(self.nnue2score * self.weight_scale_out)
            .round()
            .to(torch.int32)
        )

        callback(bias, weight, psqt_weight)

        return bias, weight, psqt_weight

    def dequantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = bias.divide(self.quantized_one)
        weight = weight.divide(self.quantized_one)
        psqt_weight = psqt_weight.divide(self.nnue2score * self.weight_scale_out)

        return bias, weight, psqt_weight

    def quantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        output_layer: bool = False,
        callback: Callable = lambda *args, **kwargs: None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kWeightScaleHidden = self.weight_scale_hidden
        kWeightScaleOut = self.nnue2score * self.weight_scale_out / self.quantized_one
        kWeightScale = kWeightScaleOut if output_layer else kWeightScaleHidden
        kBiasScaleOut = self.weight_scale_out * self.nnue2score
        kBiasScaleHidden = self.weight_scale_hidden * self.quantized_one
        kBiasScale = kBiasScaleOut if output_layer else kBiasScaleHidden
        kMaxWeight = self.quantized_one / kWeightScale

        bias = bias.mul(kBiasScale).round().to(torch.int32)

        clipped = torch.count_nonzero(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
        total_elements = torch.numel(weight)
        clipped_max = torch.max(
            torch.abs(weight.clamp(-kMaxWeight, kMaxWeight) - weight)
        )

        weight = (
            weight.clamp(-kMaxWeight, kMaxWeight)
            .mul(kWeightScale)
            .round()
            .to(torch.int8)
        )

        callback(bias, weight, clipped, total_elements, clipped_max, kMaxWeight)

        return bias, weight

    def dequantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        output_layer: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kWeightScaleHidden = self.weight_scale_hidden
        kWeightScaleOut = self.nnue2score * self.weight_scale_out / self.quantized_one
        kWeightScale = kWeightScaleOut if output_layer else kWeightScaleHidden
        kBiasScaleOut = self.weight_scale_out * self.nnue2score
        kBiasScaleHidden = self.weight_scale_hidden * self.quantized_one
        kBiasScale = kBiasScaleOut if output_layer else kBiasScaleHidden

        bias = bias.divide(kBiasScale)
        weight = weight.divide(kWeightScale)

        return bias, weight
