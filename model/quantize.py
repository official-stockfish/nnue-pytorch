from dataclasses import dataclass
from typing import Optional, Callable, NotRequired, TypedDict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import NNUEModel

FAKE_QUANTIZE_EPS = 1e-5

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
    if num_clipped > 0:
        num_clipped_int = int(num_clipped.item())
        min = rounded_value.min().item()
        max = rounded_value.max().item()
        raise RuntimeError(f"Found {num_clipped_int} out of bounds values when converting to target dtype {target_dtype}. Min: {min}, max: {max}.")

    return quantized_value

def _fake_quantize_acts(value, act_scale):
    # Fake quantization with STE
    # Inference uses bitshift which is equivalent to rounding down (floor).
    # act_scale is in nnue-pytorch is `> 1`, inverted compared to normal literature.
    # will be slightly inaccurate unless all corrections factors are 1.0.
    value_hard = value.mul(act_scale).add(FAKE_QUANTIZE_EPS).floor().div(act_scale).detach()
    value_soft = value.detach()
    value = value_hard + (value - value_soft)

    return value

def _fake_quantize_weights(value, weight_scale):
    # Fake quantization with STE
    # In contrast to activations,
    # weights use rounding as they are
    # quantized during serialization.
    value_hard = value.mul(weight_scale).round().div(weight_scale).detach()
    value_soft = value.detach()
    value = value_hard + (value - value_soft)

    return value

@dataclass
class QuantizationConfig:
    nnue2score: float = 600.0
    weight_scale_l1: float = 128.0
    weight_scale_l2: float = 64.0
    weight_scale_l_out: float = 128
    weight_scale_out: float = 16.0
    weight_quantized_max_hidden: float = 127.0 # i8 max
    ft_quantized_one: float = 256.0
    ft_quantized_max: float = 255.0 # limited to 255 for safe squaring within i16
    hidden_quantized_one: float = 128.0
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

        self.weight_scales_dict = {
            "ft_weight" : self.ft_quantized_one,
            "ft_bias" : self.ft_quantized_one,
            "ft_psqt_weight" : self.nnue2score * self.weight_scale_out,
            "ls_l1_weight" : self.weight_scale_hidden[0],
            "ls_l1_bias" : self.weight_scale_hidden[0] * self.hidden_quantized_one,
            "ls_l2_weight" : self.weight_scale_hidden[1],
            "ls_l2_bias" : self.weight_scale_hidden[1] * self.hidden_quantized_one,
            "ls_output_weight" : self.weight_scale_hidden[2],
            "ls_output_bias" : self.weight_scale_hidden[2] * self.hidden_quantized_one,
        }

    def clip_ft_act(self, preact):
        return torch.clamp(preact, 0.0, self.max_ft_activation)

    def clip_ls_act(self, preact):
        return torch.clamp(preact, 0, self.max_hidden_activation)

    def fake_quantize_ft_act(self, preact):
        act_scale = self.config.hidden_quantized_one
        return _fake_quantize_acts(preact, act_scale)

    def fake_quantize_ls_act(self, preact):
        act_scale = self.config.hidden_quantized_one
        return _fake_quantize_acts(preact, act_scale)

    def fake_quantize_skip_act(self, preact):
        # currently no separate quantization necessary, but might be necessary in the future if quant schemes change.
        return preact

    def fake_quantize_output(self, preact: torch.Tensor) -> torch.Tensor:
        multiplier_int = int(self.config.nnue2score * self.config.weight_scale_out)
        denominator_int = int(self.config.hidden_quantized_one * self.config.weight_scale_l_out * 2.0)

        fwd_out_int = torch.round(preact * denominator_int).to(torch.int64)

        output_value_int = torch.div(
            fwd_out_int * multiplier_int,
            denominator_int,
            rounding_mode='trunc'
        )

        quantized_out = output_value_int.to(preact.dtype) / float(multiplier_int)

        return quantized_out.detach() + (preact - preact.detach())

    def fake_quantize_weights(self, tensor: torch.Tensor, key: str):
        weight_scale = self.weight_scales_dict[key]
        return _fake_quantize_weights(tensor, weight_scale)

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

    def quantize_feature_transformer_weights(
        self,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
        f_weight_export_dtype: torch.dtype = torch.int16,
        callback: Optional[Callable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = weight.mul(self.weight_scales_dict["ft_weight"])
        weight = _safe_convert(weight, f_weight_export_dtype)
        psqt_weight = psqt_weight.mul(self.weight_scales_dict["ft_psqt_weight"])
        psqt_weight = _safe_convert(psqt_weight, torch.int32)

        if callback is not None:
            callback("ft_weight", weight)
            callback("psqt_weight", psqt_weight)

        return weight, psqt_weight

    def quantize_feature_transformer_bias(
        self,
        bias: torch.Tensor,
        callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        bias = bias.mul(self.weight_scales_dict["ft_bias"])
        bias = _safe_convert(bias, torch.int16)

        if callback is not None:
            callback("ft_bias", bias)

        return bias

    def dequantize_feature_transformer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        psqt_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = bias.divide(self.weight_scales_dict["ft_bias"])
        weight = weight.divide(self.weight_scales_dict["ft_weight"])
        psqt_weight = psqt_weight.divide(self.weight_scales_dict["ft_psqt_weight"])

        return bias, weight, psqt_weight

    def quantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        layer_key: str,
        callback: Optional[Callable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_key = f"{layer_key}_weight"
        bias_key = f"{layer_key}_bias"

        bias = _safe_convert(bias.mul(self.weight_scales_dict[bias_key]), torch.int32)
        weight = _safe_convert(weight.mul(self.weight_scales_dict[weight_key]), torch.int8)

        if callback is not None:
            callback(bias_key, bias)
            callback(weight_key, weight)

        return bias, weight

    def dequantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        layer_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_key = f"{layer_key}_weight"
        bias_key = f"{layer_key}_bias"

        bias = bias.divide(self.weight_scales_dict[bias_key])
        weight = weight.divide(self.weight_scales_dict[weight_key])

        return bias, weight
