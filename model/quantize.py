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

def _fake_quantize(value, act_scale):
    # Fake quantization with STE
    # Inference uses bitshift which is equivalent to rounding down (floor).
    # act_scale is in nnue-pytorch is `> 1`, inverted compared to normal literature.
    # will be slightly inaccurate unless all corrections factors are 1.0.
    value_hard = ((value * act_scale).floor() / act_scale).detach()
    value_soft = value.detach()
    value = value_hard + (value - value_soft)

    return value


@dataclass
class QuantizationConfig:
    nnue2score: float = 600.0
    weight_scale_l1: float = 64.0 # TODO 128 is better empirically for this layer
    weight_scale_l2: float = 64.0
    # weight_scale_out = (self.nnue2score * self.weight_scale_out) / self.hidden_quantized_one
    weight_scale_l_out: float = (600.0 * 16) / 128 # TODO 128 is better empirically for this layer
    weight_scale_out: float = 16.0 # TODO do calculation conversion on inference side
    weight_quantized_max_hidden: float = 127.0 # i8 max
    ft_quantized_one: float = 255.0 # TODO 255 is easier and does not require any adjustment factor
    ft_quantized_max: float = 255.0 # limited to 255 for safe squaring within i16
    hidden_quantized_one: float = 127.0 # TODO 128 is easier and does not require any adjustment factor
    hidden_quantized_max: float = 127.0 # i8 max

    # used to calculate correction factors
    inference_l0_division_factor: float = 512.0
    inference_sqr_crelu_division_factor: float = 128.0


class QuantizationManager:
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.nnue2score = config.nnue2score
        self.weight_scale_hidden = [
            config.weight_scale_l1,
            config.weight_scale_l2,
            config.weight_scale_l_out,
        ]
        self.weight_scale_out = config.weight_scale_out
        self.weight_quantized_max_hidden = config.weight_quantized_max_hidden
        self.hidden_quantized_one = config.hidden_quantized_one
        self.ft_quantized_one = config.ft_quantized_one

        hidden_q_max = config.weight_quantized_max_hidden
        self.max_hidden_weight = [hidden_q_max / scale for scale in self.weight_scale_hidden]
        # Threat weights are treated separately. A bit hacky...
        # Threat weights are quantized to int8 after scaling by ft_quantized_one
        _i8 = torch.iinfo(torch.int8)
        self.min_threat_weight = -_i8.max / config.ft_quantized_one  # -127/256
        self.max_threat_weight = _i8.max / config.ft_quantized_one  # 127/256

        self.l0_correction_factor = config.ft_quantized_one ** 2 / config.inference_l0_division_factor / self.hidden_quantized_one
        self.sqr_crelu_correction_factor = config.hidden_quantized_one / config.inference_sqr_crelu_division_factor
        self.max_ft_activation = config.ft_quantized_max / config.ft_quantized_one
        self.max_hidden_activation = config.hidden_quantized_max / config.hidden_quantized_one

    def clip_ft_act(self, preact):
        return torch.clamp(preact, 0.0, self.max_ft_activation)

    def fake_quantize_ft_act(self, preact):
        act_scale = self.config.ft_quantized_one
        return _fake_quantize(preact, act_scale)

    def clip_ls_act(self, preact):
        return torch.clamp(preact, 0, self.max_hidden_activation)

    def fake_quantize_ls_act(self, preact):
        act_scale = self.config.hidden_quantized_one
        return _fake_quantize(preact, act_scale)

    def generate_weight_clipping_config(
        self, model: "NNUEModel"
    ) -> list[WeightClippingConfig]:
        return [
            {
                "params": [model.layer_stacks.l1.linear.weight],
                "min_weight": -self.max_hidden_weight[0],
                "max_weight": self.max_hidden_weight[0],
                "virtual_params": model.layer_stacks.l1.factorized_linear.weight,
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
        ]

    def quantize_feature_transformer(
        self,
        bias: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        psqt_weight: Optional[torch.Tensor],
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
