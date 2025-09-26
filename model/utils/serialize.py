from functools import reduce
import operator
import struct
from typing import BinaryIO, Sequence

import numpy as np
import numpy.typing as npt
from numba import njit
import torch
from torch import nn


from .coalesce_weights import coalesce_ft_weights
from ..config import ModelConfig
from ..features import FeatureSet
from ..feature_transformer import BaseFeatureTransformerSlice
from ..model import NNUEModel


def ascii_hist(name, x, bins=6):
    N, X = np.histogram(x, bins=bins)
    width = 50
    nmax = N.max()

    print(name)
    for xi, n in zip(X, N):
        bar = "#" * int(n * 1.0 * width / nmax)
        xi = "{0: <8.4g}".format(xi).ljust(10)
        print("{0}| {1}".format(xi, bar))


@njit
def encode_leb_128_array(arr: npt.NDArray) -> list:
    res = []
    for v in arr:
        while True:
            byte = v & 0x7F
            v = v >> 7
            if (v == 0 and byte & 0x40 == 0) or (v == -1 and byte & 0x40 != 0):
                res.append(byte)
                break
            res.append(byte | 0x80)
    return res


@njit
def decode_leb_128_array(arr: bytes, n: int) -> npt.NDArray:
    ints = np.zeros(n)
    k = 0
    for i in range(n):
        r = 0
        shift = 0
        while True:
            byte = arr[k]
            k = k + 1
            r |= (byte & 0x7F) << shift
            shift += 7
            if (byte & 0x80) == 0:
                ints[i] = r if (byte & 0x40) == 0 else r | ~((1 << shift) - 1)
                break
    return ints


# hardcoded for now
VERSION = 0x7AF32F20
DEFAULT_DESCRIPTION = "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer."


class NNUEWriter:
    """
    All values are stored in little endian.
    """

    def __init__(
        self,
        model: NNUEModel,
        description: str | None = None,
        ft_compression: str = "none",
    ):
        if description is None:
            description = DEFAULT_DESCRIPTION

        self.buf = bytearray()

        # NOTE: model.clip_weights() should probably be called here. It's not necessary now
        # because it doesn't have more restrictive bounds than these defined by quantization,
        # but it might be necessary in the future.
        fc_hash = self.fc_hash(model)
        self.write_header(model, fc_hash, description)
        self.int32(model.feature_set.hash ^ (model.L1 * 2))  # Feature transformer hash
        self.write_feature_transformer(model, ft_compression)
        for l1, l2, output in model.layer_stacks.get_coalesced_layer_stacks():
            self.int32(fc_hash)  # FC layers hash
            self.write_fc_layer(model, l1)
            self.write_fc_layer(model, l2)
            self.write_fc_layer(model, output, is_output=True)

    @staticmethod
    def fc_hash(model: NNUEModel) -> int:
        # InputSlice hash
        prev_hash = 0xEC42E90D
        prev_hash ^= model.L1 * 2

        # Fully connected layers
        layers = [
            model.layer_stacks.l1,
            model.layer_stacks.l2,
            model.layer_stacks.output,
        ]
        for layer in layers:
            layer_hash = 0xCC03DAE4
            layer_hash += layer.out_features // model.num_ls_buckets
            layer_hash ^= prev_hash >> 1
            layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
            if layer.out_features // model.num_ls_buckets != 1:
                # Clipped ReLU hash
                layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
            prev_hash = layer_hash
        return layer_hash

    def write_header(self, model: NNUEModel, fc_hash: int, description: str) -> None:
        self.int32(VERSION)  # version
        self.int32(
            fc_hash ^ model.feature_set.hash ^ (model.L1 * 2)
        )  # halfkp network hash
        encoded_description = description.encode("utf-8")
        self.int32(len(encoded_description))  # Network definition
        self.buf.extend(encoded_description)

    def write_leb_128_array(self, arr: npt.NDArray) -> None:
        buf = encode_leb_128_array(arr)
        self.int32(len(buf))
        self.buf.extend(buf)

    def write_tensor(self, arr: npt.NDArray, compression="none") -> None:
        if compression == "none":
            self.buf.extend(arr.tobytes())
        elif compression == "leb128":
            self.buf.extend("COMPRESSED_LEB128".encode("utf-8"))
            self.write_leb_128_array(arr)
        else:
            raise Exception("Invalid compression method.")

    def write_feature_transformer(self, model: NNUEModel, ft_compression: str) -> None:
        layer = model.input

        bias = layer.bias.data[: model.L1]
        bias = bias.mul(model.quantized_one).round().to(torch.int16)

        all_weight = coalesce_ft_weights(model, layer)
        weight = all_weight[:, : model.L1]
        psqt_weight = all_weight[:, model.L1 :]

        weight = weight.mul(model.quantized_one).round().to(torch.int16)
        psqt_weight = (
            psqt_weight.mul(model.nnue2score * model.weight_scale_out)
            .round()
            .to(torch.int32)
        )

        ascii_hist("ft bias:", bias.numpy())
        ascii_hist("ft weight:", weight.numpy())
        ascii_hist("ft psqt weight:", psqt_weight.numpy())

        # Weights stored as [num_features][outputs]

        self.write_tensor(bias.flatten().numpy(), ft_compression)
        self.write_tensor(weight.flatten().numpy(), ft_compression)
        self.write_tensor(psqt_weight.flatten().numpy(), ft_compression)

    def write_fc_layer(
        self, model: NNUEModel, layer: nn.Linear, is_output=False
    ) -> None:
        # FC layers are stored as int8 weights, and int32 biases
        kWeightScaleHidden = model.weight_scale_hidden
        kWeightScaleOut = (
            model.nnue2score * model.weight_scale_out / model.quantized_one
        )
        kWeightScale = kWeightScaleOut if is_output else kWeightScaleHidden
        kBiasScaleOut = model.weight_scale_out * model.nnue2score
        kBiasScaleHidden = model.weight_scale_hidden * model.quantized_one
        kBiasScale = kBiasScaleOut if is_output else kBiasScaleHidden
        kMaxWeight = model.quantized_one / kWeightScale

        bias = layer.bias.data
        bias = bias.mul(kBiasScale).round().to(torch.int32)

        weight = layer.weight.data
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

        ascii_hist("fc bias:", bias.numpy())
        print(
            "layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(
                clipped, total_elements, clipped_max, kMaxWeight
            )
        )
        ascii_hist("fc weight:", weight.numpy())

        # FC inputs are padded to 32 elements by spec.
        num_input = weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
            new_w[:, : weight.shape[1]] = weight
            weight = new_w

        self.buf.extend(bias.flatten().numpy().tobytes())
        # Weights stored as [outputs][inputs], so we can flatten
        self.buf.extend(weight.flatten().numpy().tobytes())

    def int32(self, v: int) -> None:
        self.buf.extend(struct.pack("<I", v))


class NNUEReader:
    def __init__(self, f: BinaryIO, feature_set: FeatureSet, config: ModelConfig):
        self.f = f
        self.feature_set = feature_set
        self.model = NNUEModel(feature_set, config)
        self.config = config
        fc_hash = NNUEWriter.fc_hash(self.model)

        self.read_header(feature_set, fc_hash)
        self.read_int32(
            feature_set.hash ^ (self.config.L1 * 2)
        )  # Feature transformer hash
        self.read_feature_transformer(self.model.input, self.model.num_psqt_buckets)
        for i in range(self.model.num_ls_buckets):
            l1 = nn.Linear(2 * self.config.L1 // 2, self.config.L2 + 1)
            l2 = nn.Linear(self.config.L2 * 2, self.config.L3)
            output = nn.Linear(self.config.L3, 1)

            self.read_int32(fc_hash)  # FC layers hash
            self.read_fc_layer(l1)
            self.read_fc_layer(l2)
            self.read_fc_layer(output, is_output=True)

            self.model.layer_stacks.l1.weight.data[
                i * (self.config.L2 + 1) : (i + 1) * (self.config.L2 + 1), :
            ] = l1.weight
            self.model.layer_stacks.l1.bias.data[
                i * (self.config.L2 + 1) : (i + 1) * (self.config.L2 + 1)
            ] = l1.bias
            self.model.layer_stacks.l2.weight.data[
                i * self.config.L3 : (i + 1) * self.config.L3, :
            ] = l2.weight
            self.model.layer_stacks.l2.bias.data[
                i * self.config.L3 : (i + 1) * self.config.L3
            ] = l2.bias
            self.model.layer_stacks.output.weight.data[i : (i + 1), :] = output.weight
            self.model.layer_stacks.output.bias.data[i : (i + 1)] = output.bias

    def read_header(self, feature_set: FeatureSet, fc_hash: int) -> None:
        self.read_int32(VERSION)  # version
        self.read_int32(fc_hash ^ feature_set.hash ^ (self.config.L1 * 2))
        desc_len = self.read_int32()
        self.description = self.f.read(desc_len).decode("utf-8")

    def read_leb_128_array(
        self, dtype: npt.DTypeLike, shape: Sequence[int]
    ) -> torch.Tensor:
        len_bytes = self.read_int32()
        d = self.f.read(len_bytes)
        if len(d) != len_bytes:
            raise Exception("Unexpected end of file when reading compressed data.")

        res = torch.tensor(decode_leb_128_array(d, reduce(operator.mul, shape, 1)))
        res = res.reshape(shape)
        return res

    def peek(self, length: int = 1) -> bytes:
        pos = self.f.tell()
        data = self.f.read(length)
        self.f.seek(pos)
        return data

    def determine_compression(self) -> str:
        leb128_magic = b"COMPRESSED_LEB128"
        if self.peek(len(leb128_magic)) == leb128_magic:
            self.f.read(len(leb128_magic))  # actually advance the file pointer
            return "leb128"
        else:
            return "none"

    def tensor(self, dtype: npt.DTypeLike, shape: Sequence[int]) -> torch.Tensor:
        compression = self.determine_compression()

        if compression == "none":
            d = np.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
            d = torch.from_numpy(d.astype(np.float32))
            d = d.reshape(shape)
            return d
        elif compression == "leb128":
            return self.read_leb_128_array(dtype, shape)
        else:
            raise Exception("Invalid compression method.")

    def read_feature_transformer(
        self, layer: BaseFeatureTransformerSlice, num_psqt_buckets: int
    ) -> None:
        shape = layer.weight.shape

        bias = self.tensor(np.int16, [layer.bias.shape[0] - num_psqt_buckets]).divide(
            self.model.quantized_one
        )
        # weights stored as [num_features][outputs]
        weights = self.tensor(np.int16, [shape[0], shape[1] - num_psqt_buckets])
        weights = weights.divide(self.model.quantized_one)
        psqt_weights = self.tensor(np.int32, [shape[0], num_psqt_buckets])
        psqt_weights = psqt_weights.divide(
            self.model.nnue2score * self.model.weight_scale_out
        )

        layer.bias.data = torch.cat([bias, torch.tensor([0] * num_psqt_buckets)])
        layer.weight.data = torch.cat([weights, psqt_weights], dim=1)

    def read_fc_layer(self, layer: nn.Linear, is_output: bool = False) -> None:
        kWeightScaleHidden = self.model.weight_scale_hidden
        kWeightScaleOut = (
            self.model.nnue2score
            * self.model.weight_scale_out
            / self.model.quantized_one
        )
        kWeightScale = kWeightScaleOut if is_output else kWeightScaleHidden
        kBiasScaleOut = self.model.weight_scale_out * self.model.nnue2score
        kBiasScaleHidden = self.model.weight_scale_hidden * self.model.quantized_one
        kBiasScale = kBiasScaleOut if is_output else kBiasScaleHidden

        # FC inputs are padded to 32 elements by spec.
        non_padded_shape = layer.weight.shape
        padded_shape = (non_padded_shape[0], ((non_padded_shape[1] + 31) // 32) * 32)

        layer.bias.data = self.tensor(np.int32, layer.bias.shape).divide(kBiasScale)
        layer.weight.data = self.tensor(np.int8, padded_shape).divide(kWeightScale)

        # Strip padding.
        layer.weight.data = layer.weight.data[
            : non_padded_shape[0], : non_padded_shape[1]
        ]

    def read_int32(self, expected: int | None = None) -> int:
        v = struct.unpack("<I", self.f.read(4))[0]
        if expected is not None and v != expected:
            raise Exception("Expected: %x, got %x" % (expected, v))
        return v
