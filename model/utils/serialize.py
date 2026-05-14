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


def ascii_hist(name, x, bins=7):
    start, end = int(x.min()), int(x.max())
    if start >= end - bins:
        start -= (bins + 1) // 2
        end += bins // 2
    edges = np.linspace(start, end + 1, bins + 1).astype(int)
    edges = np.unique(edges)
    N, X = np.histogram(x, bins=edges)
    width = 50
    nmax = N.max()

    print(name)
    for xi, n in zip(X, N):
        bar = "#" * int(n * 1.0 * width / nmax)
        xi = "{0: <8.4g}".format(xi).ljust(10)
        print("{0}| {1}".format(xi, bar))

def get_histogram_callback(hist_title: str, verbose: bool):
    if not verbose:
        return None

    def histogram_callback(
        hist_subtitle: str,
        values: torch.Tensor,
    ):
        total_elements = values.numel()
        hist_desc = [hist_title, hist_subtitle]
        hist_desc = " ".join(filter(None, hist_desc))

        if total_elements == 0:
            print(f"Layer '{hist_desc}' is empty.")
            return

        min_value = values.min().item()
        num_argmin = int((values == min_value).sum().item())
        max_value = values.max().item()
        num_argmax = int((values == max_value).sum().item())

        ascii_hist(f"{hist_desc}: ", values.detach().cpu().numpy())
        print(
            f"Minimum value in layer is {min_value}, occurring {num_argmin} times.\n"
            f"Maximum value in layer is {max_value}, occurring {num_argmax} times."
        )

    return histogram_callback

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
VERSION = 0x6A448AFA
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
        verbose: bool = True,
    ):
        if description is None:
            description = DEFAULT_DESCRIPTION

        self.buf = bytearray()
        self.verbose = verbose

        fc_hash = self.fc_hash(model)
        self.write_header(model, fc_hash, description)
        self.int32(model.feature_hash ^ (model.L1 * 2))  # Feature transformer hash
        self.write_feature_transformer(model, ft_compression)
        for bucket, (l1, l2, output) in enumerate(model.layer_stacks.get_coalesced_layer_stacks()):
            self.int32(fc_hash)  # FC layers hash
            self.write_fc_layer(model, l1, 0, f"bucket {bucket} l1")
            self.write_fc_layer(model, l2, 1, f"bucket {bucket} l2")
            self.write_fc_layer(model, output, 2, f"bucket {bucket} output")

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

    def write_tensor(self, arr: torch.Tensor, compression="none") -> None:
        arr = arr.detach().flatten().cpu().numpy()
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

        # biases are exported as i16s
        biases, _, _ = model.quantization.quantize_feature_transformer(
            bias, None, None, torch.int16, get_histogram_callback("", self.verbose)
        )

        self.write_tensor(biases, ft_compression)

        # Weights stored as [num_features][outputs]
        offset = 0
        for f in layer.features:
            n = f.NUM_REAL_FEATURES
            f_export_dtype = f.EXPORT_WEIGHT_DTYPE

            ft_histogram_callback = get_histogram_callback(f.FEATURE_NAME, self.verbose)
            segment_weight = weight[offset : offset + n]
            segment_psqt_weight = psqt_weight[offset : offset + n]
            _, segment_weight, segment_psqt_weight = model.quantization.quantize_feature_transformer(
                None, segment_weight, segment_psqt_weight, f_export_dtype, ft_histogram_callback
            )
            # threat weights are expected to always be uncompressed -- should be changed in the future
            segment_compression = ft_compression if not f_export_dtype == torch.int8 else "none"
            offset += n

            self.write_tensor(segment_weight, segment_compression)
            self.write_tensor(segment_psqt_weight, ft_compression)


    def write_fc_layer(
        self,
        model: NNUEModel,
        layer: nn.Linear,
        layer_idx: int,
        desc: str,
    ) -> None:
        # FC layers are stored as int8 weights, and int32 biases
        bias = layer.bias.data
        weight = layer.weight.data

        bias, weight = model.quantization.quantize_fc_layer(
            bias, weight, layer_idx, get_histogram_callback(desc, self.verbose)
        )

        # FC inputs are padded to 32 elements by spec.
        num_input = weight.shape[1]
        if num_input % 32 != 0:
            num_input += 32 - (num_input % 32)
            new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
            new_w[:, : weight.shape[1]] = weight
            weight = new_w

        self.write_tensor(bias, "none")
        # Weights stored as [outputs][inputs], so we can flatten
        self.write_tensor(weight, "none")

    def int32(self, v: int) -> None:
        self.buf.extend(struct.pack("<I", v))


class NNUEReader:
    def __init__(
        self,
        f: BinaryIO,
        feature_name: str,
        config: ModelConfig,
    ):
        self.f = f
        self.feature_name = feature_name
        self.model = NNUEModel(feature_name, config)
        self.config = config
        fc_hash = NNUEWriter.fc_hash(self.model)

        self.read_header(self.model.feature_hash, fc_hash)
        self.read_int32(
            self.model.feature_hash ^ (self.config.L1 * 2)
        )  # Feature transformer hash
        self.read_feature_transformer(self.model.input, self.model.num_psqt_buckets)

        layers = [
            self.model.layer_stacks.l1,
            self.model.layer_stacks.l2,
            self.model.layer_stacks.output,
        ]
        num_ls_buckets = self.model.num_ls_buckets
        l_w_slices = [
            torch.chunk(layer.linear.weight.data, num_ls_buckets, dim=0)
            for layer in layers
        ]
        l_b_slices = [
            torch.chunk(layer.linear.bias.data, num_ls_buckets, dim=0)
            for layer in layers
        ]

        for b in range(num_ls_buckets):
            self.read_int32(fc_hash)  # FC layers hash
            for layer_idx in range(len(layers)):
                self.read_fc_layer(
                    l_w_slices[layer_idx][b],
                    l_b_slices[layer_idx][b],
                    layer_idx,
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

    def read_feature_transformer(self, layer, num_psqt_buckets: int) -> None:
        num_outputs = layer.num_outputs
        L1 = num_outputs - num_psqt_buckets

        bias = self.tensor(np.int16, [L1])
        segments = []
        segments_psqt = []

        for feature in layer.features:
            dtype = np.int8 if feature.EXPORT_WEIGHT_DTYPE == torch.int8 else np.int16
            s = self.tensor(dtype, [feature.NUM_REAL_FEATURES, L1])
            segments.append(s)
            s_psqt = self.tensor(np.int32, [feature.NUM_REAL_FEATURES, num_psqt_buckets])
            segments_psqt.append(s_psqt)

        weight = torch.cat(segments, dim=0)
        psqt_weight = torch.cat(segments_psqt, dim=0)

        bias, weight, psqt_weight = (
            self.model.quantization.dequantize_feature_transformer(
                bias, weight, psqt_weight
            )
        )

        # Combine weight and psqt_weight into export format, then expand
        export_weight = torch.cat([weight, psqt_weight], dim=1)
        layer.load_export_weights(export_weight)
        layer.bias.data = torch.cat([bias, torch.tensor([0] * num_psqt_buckets)])

    def read_fc_layer(
        self,
        layer_weight_t: torch.Tensor,
        layer_bias_t: torch.Tensor,
        layer_idx: int,
    ) -> None:
        # FC inputs are padded to 32 elements by spec.
        non_padded_shape = layer_weight_t.shape
        padded_shape = (non_padded_shape[0], ((non_padded_shape[1] + 31) // 32) * 32)

        bias = self.tensor(np.int32, layer_bias_t.shape)
        weight = self.tensor(np.int8, padded_shape)

        bias, weight = self.model.quantization.dequantize_fc_layer(
            bias, weight, layer_idx
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
