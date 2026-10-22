from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch

if TYPE_CHECKING:
    from .model import NNUEModel

FAKE_QUANTIZE_EPS = 1e-5
INT8_MAX = float(torch.iinfo(torch.int8).max)


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
        raise RuntimeError(
            f"Found {num_clipped_int} out of bounds values when converting to target dtype {target_dtype}. Min: {min}, max: {max}."
        )

    return quantized_value


def _fake_quantize_acts(value, act_scale):
    # Fake quantization with STE
    # Inference uses bitshift which is equivalent to rounding down (floor).
    # act_scale is in nnue-pytorch is `> 1`, inverted compared to normal literature.
    # will be slightly inaccurate unless all corrections factors are 1.0.
    value_hard = (
        value.mul(act_scale).add(FAKE_QUANTIZE_EPS).floor().div(act_scale).detach()
    )
    value_soft = value.detach()
    value = value_hard + (value - value_soft)

    return value


def _fake_quantize_weights(value: torch.Tensor, weight_scale: float) -> torch.Tensor:
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
    ft_quantized_one: float = 256.0
    ft_quantized_max: float = 255.0  # limited to 255 for safe squaring within i16
    hidden_quantized_one: float = 128.0
    hidden_quantized_max: float = 127.0  # i8 max

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
        self.hidden_quantized_one = config.hidden_quantized_one
        self.ft_quantized_one = config.ft_quantized_one

        # FC layers are stored as int8 weights
        i8_max = INT8_MAX
        self.max_hidden_weight = [i8_max / scale for scale in self.weight_scale_hidden]

        self.l0_correction_factor = (
            config.ft_quantized_one**2
            / config.inference_l0_division_factor
            / self.hidden_quantized_one
        )
        self.sqr_crelu_correction_factor = (
            config.hidden_quantized_one / config.inference_sqr_crelu_division_factor
        )
        self.max_ft_activation = config.ft_quantized_max / config.ft_quantized_one
        self.max_hidden_activation = (
            config.hidden_quantized_max / config.hidden_quantized_one
        )
        self.weight_scales_dict = {
            "ft_weight": self.ft_quantized_one,
            "ft_bias": self.ft_quantized_one,
            "ls_l1_weight": self.weight_scale_hidden[0],
            "ls_l1_bias": self.weight_scale_hidden[0] * self.hidden_quantized_one,
            "ls_l2_weight": self.weight_scale_hidden[1],
            "ls_l2_bias": self.weight_scale_hidden[1] * self.hidden_quantized_one,
            "ls_output_weight": self.weight_scale_hidden[2],
            "ls_output_bias": self.weight_scale_hidden[2] * self.hidden_quantized_one,
        }

    def get_ft_weight_bounds(self, dtype: torch.dtype) -> tuple[float, float]:
        info = torch.iinfo(dtype)
        return -info.max / self.ft_quantized_one, info.max / self.ft_quantized_one

    def get_ls_weight_bounds(self, layer_key: str) -> tuple[float, float]:
        scale_key = (
            layer_key if layer_key.endswith("_weight") else f"{layer_key}_weight"
        )
        scale = self.weight_scales_dict[scale_key]
        max_w = INT8_MAX / scale
        return -max_w, max_w

    def clip_ls_weight(self, weight: torch.Tensor, layer_key: str) -> torch.Tensor:
        min_w, max_w = self.get_ls_weight_bounds(layer_key)
        return weight.clamp(min_w, max_w)

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
        denominator_int = int(
            self.config.hidden_quantized_one * self.config.weight_scale_l_out * 2.0
        )

        fwd_out_int = torch.round(preact * denominator_int).to(torch.int64)

        output_value_int = torch.div(
            fwd_out_int * multiplier_int, denominator_int, rounding_mode="trunc"
        )

        quantized_out = output_value_int.to(preact.dtype) / float(multiplier_int)

        return quantized_out.detach() + (preact - preact.detach())

    def fake_quantize_weights(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        weight_scale = self.weight_scales_dict[key]
        return _fake_quantize_weights(tensor, weight_scale)

    def generate_weight_clipping_config(
        self, model: "NNUEModel"
    ) -> list[WeightClippingConfig]:
        return []

    def quantize_feature_transformer_weights(
        self,
        weight: torch.Tensor,
        f_weight_export_dtype: torch.dtype = torch.int16,
        callback: Callable | None = None,
    ) -> torch.Tensor:
        weight = weight.mul(self.weight_scales_dict["ft_weight"])
        weight = _safe_convert(weight, f_weight_export_dtype)

        if callback is not None:
            callback("ft_weight", weight)

        return weight

    def quantize_feature_transformer_bias(
        self,
        bias: torch.Tensor,
        callback: Callable | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bias = bias.divide(self.weight_scales_dict["ft_bias"])
        weight = weight.divide(self.weight_scales_dict["ft_weight"])

        return bias, weight

    def quantize_fc_layer(
        self,
        bias: torch.Tensor,
        weight: torch.Tensor,
        layer_key: str,
        callback: Callable | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_key = f"{layer_key}_weight"
        bias_key = f"{layer_key}_bias"

        bias = _safe_convert(bias.mul(self.weight_scales_dict[bias_key]), torch.int32)
        weight = _safe_convert(
            weight.mul(self.weight_scales_dict[weight_key]), torch.int8
        )

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

    def get_dynamic_bias_scales(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        s1 = self.weight_scales_dict["ls_l1_bias"]
        s2 = self.weight_scales_dict["ls_l2_bias"]
        return torch.cat(
            [
                torch.full((64,), s1, device=device, dtype=dtype),
                torch.full((64,), s2, device=device, dtype=dtype),
            ]
        )

    def quantize_dynamic_bias(
        self,
        biases: torch.Tensor,
        callback: Callable | None = None,
    ) -> torch.Tensor:
        scales = self.get_dynamic_bias_scales(device=biases.device, dtype=biases.dtype)
        q_biases = _safe_convert(biases.mul(scales), torch.int32)
        if callback is not None:
            callback("dynamic_bias", q_biases)
        return q_biases

    def dequantize_dynamic_bias(self, biases: torch.Tensor) -> torch.Tensor:
        scales = self.get_dynamic_bias_scales(device=biases.device, dtype=biases.dtype)
        return biases.divide(scales)

    def fake_quantize_dynamic_bias(self, biases: torch.Tensor) -> torch.Tensor:
        scales = self.get_dynamic_bias_scales(device=biases.device, dtype=biases.dtype)
        biases_hard = biases.mul(scales).round().div(scales).detach()
        biases_soft = biases.detach()
        return biases_hard + (biases - biases_soft)
