from functools import reduce
import operator
import struct
from typing import BinaryIO, Sequence

import numpy as np
import numpy.typing as npt
from numba import njit
import torch
from torch import nn

from ..config import ModelConfig
from ..model import NNUEModel
from ..modules import BaseFeatureTransformer
from ..modules.features import get_feature_cls
from ..modules.features.full_threats import FullThreats
from ..quantize import QuantizationConfig


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

        fc_hash = self.fc_hash(model)
        self.write_header(model, fc_hash, description)
        self.int32(model.feature_hash ^ (model.L1 * 2))  # Feature transformer hash
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
            model.layer_stacks.l1.linear,
            model.layer_stacks.l2.linear,
            model.layer_stacks.output.linear,
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
        self.int32(fc_hash ^ model.feature_hash ^ (model.L1 * 2))  # halfkp network hash
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

        # Get export weights (coalesced + remapped 12→11 piece types)
        export_weight = layer.get_export_weights()
        weight = export_weight[:, : model.L1]
        psqt_weight = export_weight[:, model.L1 :]

        def histogram_callback(
            bias: torch.Tensor, weight: torch.Tensor, psqt_weight: torch.Tensor
        ):
            ascii_hist("ft bias:", bias.numpy())
            ascii_hist("ft weight:", weight.numpy())
            ascii_hist("ft psqt weight:", psqt_weight.numpy())

        bias, weight, psqt_weight = model.quantization.quantize_feature_transformer(
            bias, weight, psqt_weight, histogram_callback
        )

        # Weights stored as [num_features][outputs]
        self.write_tensor(bias.flatten().numpy(), ft_compression)
        if isinstance(layer, FullThreats):
            threat_weight = weight[: layer.NUM_THREAT_FEATURES].to(torch.int8)
            psq_weight = weight[layer.NUM_THREAT_FEATURES :]
            self.write_tensor(threat_weight.flatten().numpy())
            self.write_tensor(psq_weight.flatten().numpy(), ft_compression)
        else:
            self.write_tensor(weight.flatten().numpy(), ft_compression)
        self.write_tensor(psqt_weight.flatten().numpy(), ft_compression)

    def write_fc_layer(
        self, model: NNUEModel, layer: nn.Linear, is_output=False
    ) -> None:
        # FC layers are stored as int8 weights, and int32 biases
        bias = layer.bias.data
        weight = layer.weight.data

        def histogram_callback(
            bias: torch.Tensor,
            weight: torch.Tensor,
            clipped: torch.Tensor,
            total_elements: int,
            clipped_max: torch.Tensor,
            kMaxWeight: float,
        ):
            ascii_hist("fc bias:", bias.numpy())
            print(
                "layer has {}/{} clipped weights. Exceeding by {} the maximum {}.".format(
                    clipped, total_elements, clipped_max, kMaxWeight
                )
            )
            ascii_hist("fc weight:", weight.numpy())

        bias, weight = model.quantization.quantize_fc_layer(
            bias, weight, is_output, histogram_callback
        )

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
    def __init__(
        self,
        f: BinaryIO,
        feature_name: str,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
    ):
        self.f = f
        self.feature_name = feature_name
        self.model = NNUEModel(feature_name, config, quantize_config)
        self.config = config
        fc_hash = NNUEWriter.fc_hash(self.model)

        feature_cls = get_feature_cls(feature_name)
        self.read_header(feature_cls.HASH, fc_hash)
        self.read_int32(
            feature_cls.HASH ^ (self.config.L1 * 2)
        )  # Feature transformer hash
        self.read_feature_transformer(self.model.input, self.model.num_psqt_buckets)

        layers = [
            self.model.layer_stacks.l1,
            self.model.layer_stacks.l2,
            self.model.layer_stacks.output,
        ]
        num_ls_buckets = self.model.num_ls_buckets
        l_w_slices = [
            torch.chunk(l.linear.weight.data, num_ls_buckets, dim=0) for l in layers
        ]
        l_b_slices = [
            torch.chunk(l.linear.bias.data, num_ls_buckets, dim=0) for l in layers
        ]

        for b in range(num_ls_buckets):
            self.read_int32(fc_hash)  # FC layers hash
            for l in range(len(layers)):
                self.read_fc_layer(
                    l_w_slices[l][b], l_b_slices[l][b], is_output=(l == len(layers) - 1)
                )

    def read_header(self, feature_hash: int, fc_hash: int) -> None:
        self.read_int32(VERSION)  # version
        self.read_int32(fc_hash ^ feature_hash ^ (self.config.L1 * 2))
        desc_len = self.read_int32()
        self.description = self.f.read(desc_len).decode("utf-8")

    def read_leb_128_array(
        self, dtype: npt.DTypeLike, shape: Sequence[int]
    ) -> torch.Tensor:
        len_bytes = self.read_int32()
        d = self.f.read(len_bytes)
        if len(d) != len_bytes:
            raise Exception("Unexpected end of file when reading compressed data.")

        res = torch.tensor(
            decode_leb_128_array(d, reduce(operator.mul, shape, 1)), dtype=torch.float32
        )
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
        self, layer: BaseFeatureTransformer, num_psqt_buckets: int
    ) -> None:
        num_export_features = layer.NUM_REAL_FEATURES
        num_outputs = layer.num_outputs

        bias = self.tensor(np.int16, [num_outputs - num_psqt_buckets])
        # weights stored as [num_features][outputs]
        if isinstance(layer, FullThreats):
            threat_weight = self.tensor(
                np.int8, [layer.NUM_THREAT_FEATURES, num_outputs - num_psqt_buckets]
            )
            psq_weight = self.tensor(
                np.int16,
                [
                    num_export_features - layer.NUM_THREAT_FEATURES,
                    num_outputs - num_psqt_buckets,
                ],
            )
            weight = torch.cat([threat_weight, psq_weight], dim=0)
        else:
            weight = self.tensor(
                np.int16, [num_export_features, num_outputs - num_psqt_buckets]
            )
        psqt_weight = self.tensor(np.int32, [num_export_features, num_psqt_buckets])

        bias, weight, psqt_weight = (
            self.model.quantization.dequantize_feature_transformer(
                bias, weight, psqt_weight
            )
        )

        # Combine weight and psqt_weight into export format, then expand 11→12
        export_weight = torch.cat([weight, psqt_weight], dim=1)
        layer.load_export_weights(export_weight)
        layer.bias.data = torch.cat([bias, torch.tensor([0] * num_psqt_buckets)])

    def read_fc_layer(
        self,
        layer_weight_t: torch.Tensor,
        layer_bias_t: torch.Tensor,
        is_output: bool = False,
    ) -> None:
        # FC inputs are padded to 32 elements by spec.
        non_padded_shape = layer_weight_t.shape
        padded_shape = (non_padded_shape[0], ((non_padded_shape[1] + 31) // 32) * 32)

        bias = self.tensor(np.int32, layer_bias_t.shape)
        weight = self.tensor(np.int8, padded_shape)

        bias, weight = self.model.quantization.dequantize_fc_layer(
            bias, weight, is_output
        )

        layer_bias = bias
        # Strip padding.
        layer_weight = weight[: non_padded_shape[0], : non_padded_shape[1]]

        layer_weight_t.data.copy_(layer_weight)
        layer_bias_t.data.copy_(layer_bias)

    def read_int32(self, expected: int | None = None) -> int:
        v = struct.unpack("<I", self.f.read(4))[0]
        if expected is not None and v != expected:
            raise Exception("Expected: %x, got %x" % (expected, v))
        return v
