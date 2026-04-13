from dataclasses import dataclass
from typing import Optional, Callable, NotRequired, TypedDict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import NNUEModel


class WeightClippingConfig(TypedDict):
    params: list[torch.Tensor]
    min_weight: float
    max_weight: float
    virtual_params: NotRequired[torch.Tensor]

def _safe_convert(value: torch.Tensor, target_dtype: torch.dtype):
    _info = torch.iinfo(target_dtype)
    # Symmetric range: [-max, max]
    min_val = -_info.max
    max_val = _info.max

    rounded_value = value.round()
    clamped_value = rounded_value.clamp(min_val, max_val)
    num_clipped = (rounded_value != clamped_value).sum()
    quantized_value = clamped_value.to(target_dtype)
    return quantized_value, num_clipped

@dataclass
class QuantizationConfig:
    nnue2score: float = 600.0
    weight_scale_hidden_0: float = 128.0
    weight_scale_hidden_1: float = 64.0
    weight_scale_hidden_2: float = 128.0
    weight_scale_out: float = 16.0
    weight_scale_router: float = 256.0
    weight_quantized_max_hidden: float = 127.0 # i8 max
    ft_quantized_one: float = 256.0
    ft_quantized_max: float = 255.0 # limited to 255 for safe squaring within i16
    hidden_quantized_one: float = 128.0
    hidden_quantized_max: float = 127.0 # i8 max
    inference_l0_division_factor: float = 512.0
    inference_sqcrele_division_factor: float = 128.0

class QuantizationManager:
    def __init__(self, config: QuantizationConfig):
        self.nnue2score = config.nnue2score
        self.weight_scale_hidden = [
            config.weight_scale_hidden_0,
            config.weight_scale_hidden_1,
            config.weight_scale_hidden_2,
        ]
        self.weight_scale_out = config.weight_scale_out
        self.weight_scale_router = config.weight_scale_router

        self.weight_quantized_max_hidden = config.weight_quantized_max_hidden
        self.hidden_quantized_one = config.hidden_quantized_one
        self.ft_quantized_one = config.ft_quantized_one

        hidden_q_max = config.weight_quantized_max_hidden
        self.max_hidden_weight = [hidden_q_max / scale for scale in self.weight_scale_hidden]
        self.max_router_weight = hidden_q_max / config.weight_scale_router
        # Thread weights are treated separately. A bit hacky...
        # Threat weights are quantized to int8 after scaling by ft_quantized_one
        _i8 = torch.iinfo(torch.int8)
        self.min_threat_weight = -_i8.max / config.ft_quantized_one  # -127/256
        self.max_threat_weight = _i8.max / config.ft_quantized_one  # 127/256

        self._l0_correction_factor = config.ft_quantized_one ** 2 / config.inference_l0_division_factor / self.hidden_quantized_one
        self._sqcrele_correction_factor = config.hidden_quantized_one / config.inference_sqcrele_division_factor
        self._max_ft_activation = config.ft_quantized_max / config.ft_quantized_one
        self._max_hidden_activation = config.hidden_quantized_max / config.hidden_quantized_one

    @property
    def l0_correction_factor(self):
        return self._l0_correction_factor

    @property
    def sqcrele_correction_factor(self):
        return self._sqcrele_correction_factor

    @property
    def max_ft_activation(self):
        return self._max_ft_activation

    @property
    def max_hidden_activation(self):
        return self._max_hidden_activation

    def generate_weight_clipping_config(
        self, model: "NNUEModel"
    ) -> list[WeightClippingConfig]:
        return [
            {
                "params": [model.layer_stacks.l1.linear.weight],
                "min_weight": -self.max_hidden_weight[0],
                "max_weight": self.max_hidden_weight[0],
            },
            {
                "params": [model.layer_stacks.l2.linear.weight],
                "min_weight": -self.max_hidden_weight[1],
                "max_weight": self.max_hidden_weight[1],
            },
            {
                "params": [model.layer_stacks.output.linear.weight],
                "min_weight": -self.max_hidden_weight[2],
                "max_weight": self.max_hidden_weight[2],
            },
            {
                "params": [model.router.linear.weight],
                "min_weight": -self.max_hidden_weight[2],
                "max_weight": self.max_hidden_weight[2],
            },
        ]

    def quantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
        f_weight_export_dtype: torch.dtype = torch.int16,
        callback: Optional[Callable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bias is not None:
            # only weight can have different dtypes, bias is always int16, psqt_weight is always int32
            bias = bias.mul(self.ft_quantized_one)
            bias, bias_clipped = _safe_convert(bias, torch.int16)

            if callback is not None:
                callback("ft_bias", bias, bias_clipped)

        if weight is not None:
            weight = weight.mul(self.ft_quantized_one)
            weight, weight_clipped = _safe_convert(weight, f_weight_export_dtype)

            if callback is not None:
                callback("ft_weight", weight, weight_clipped)

        if psqt_weight is not None:
            psqt_weight = psqt_weight.mul(self.nnue2score * self.weight_scale_out)
            psqt_weight, psqt_weight_clipped = _safe_convert(psqt_weight, torch.int32)

            if callback is not None:
                callback("psqt_weight", psqt_weight, psqt_weight_clipped)

        return bias, weight, psqt_weight

    def dequantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = bias.divide(self.ft_quantized_one) if bias is not None else None
        weight = weight.divide(self.ft_quantized_one) if weight is not None else None
        psqt_weight = psqt_weight.divide(self.nnue2score * self.weight_scale_out) if psqt_weight is not None else None

        return bias, weight, psqt_weight

    def quantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        layer_idx: int,
        callback: Optional[Callable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kBiasScaleHidden = self.weight_scale_hidden[layer_idx] * self.hidden_quantized_one
        kWeightScaleHidden = self.weight_scale_hidden[layer_idx]

        bias, bias_clipped = _safe_convert(bias.mul(kBiasScaleHidden), torch.int32)
        weight, weight_clipped = _safe_convert(weight.mul(kWeightScaleHidden), torch.int8)

        if callback is not None:
            callback("fc_weight", weight, weight_clipped)
            callback("fc_bias", bias, bias_clipped)

        return bias, weight

    def dequantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kBiasScaleHidden = self.weight_scale_hidden[layer_idx] * self.hidden_quantized_one
        kWeightScaleHidden = self.weight_scale_hidden[layer_idx]

        bias = bias.divide(kBiasScaleHidden)
        weight = weight.divide(kWeightScaleHidden)

        return bias, weight

    def quantize_router(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        callback: Optional[Callable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kBiasScaleHidden = self.weight_scale_router * self.hidden_quantized_one
        kWeightScaleHidden = self.weight_scale_router

        bias, bias_clipped = _safe_convert(bias.mul(kBiasScaleHidden), torch.int32)
        weight, weight_clipped = _safe_convert(weight.mul(kWeightScaleHidden), torch.int8)

        if callback is not None:
            callback("router_weight", weight, weight_clipped)
            callback("router_bias", bias, bias_clipped)

        return bias, weight

    def dequantize_router(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kBiasScaleHidden = self.weight_scale_router * self.hidden_quantized_one
        kWeightScaleHidden = self.weight_scale_router

        bias = bias.divide(kBiasScaleHidden)
        weight = weight.divide(kWeightScaleHidden)

        return bias, weight
