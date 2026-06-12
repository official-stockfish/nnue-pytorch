import ctypes
import os
import glob

import numpy as np
import torch

from .config import CDataloaderSkipConfig, CDataloaderDDPConfig


def _pin_and_move(t: torch.Tensor, device, use_pinned_memory=False, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = t.dtype

    # Must copy off SparseBatch-backed memory before it is freed
    if torch.cuda.is_available() and use_pinned_memory:
        # Allocate a pinned CPU tensor and copy the data directly into it.
        # This is much faster than t.clone().pin_memory() which does two copies/allocations.
        out = torch.empty(t.shape, dtype=dtype, layout=t.layout, device="cpu", pin_memory=True)
        out.copy_(t)
        if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
            return out
        return out.to(device=device, non_blocking=True)
    
    # If not using pinned memory, just copy to standard CPU storage
    out = torch.empty(t.shape, dtype=dtype, layout=t.layout, device="cpu")
    out.copy_(t)
    if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
        return out
    return out.to(device=device)


class SparseBatch(ctypes.Structure):
    _fields_ = [
        ("num_inputs", ctypes.c_int),
        ("size", ctypes.c_int),
        ("is_white", ctypes.POINTER(ctypes.c_float)),
        ("outcome", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("num_active_white_features", ctypes.c_int),
        ("num_active_black_features", ctypes.c_int),
        ("max_active_features", ctypes.c_int),
        ("white", ctypes.POINTER(ctypes.c_int)),
        ("black", ctypes.POINTER(ctypes.c_int)),
        ("white_values", ctypes.POINTER(ctypes.c_float)),
        ("black_values", ctypes.POINTER(ctypes.c_float)),
        ("psqt_indices", ctypes.POINTER(ctypes.c_int)),
        ("layer_stack_indices", ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device, use_pinned_memory=False):
        total_floats = self.size * 3 + self.size * self.max_active_features * 2
        total_ints = self.size * 2 + self.size * self.max_active_features * 2

        # Create CPU-side tensors sharing the contiguous C++ buffers
        # self.is_white points to the start of the float block
        float_block_cpu = torch.from_numpy(
            np.ctypeslib.as_array(self.is_white, shape=(total_floats,))
        )
        # self.white points to the start of the int block
        int_block_cpu = torch.from_numpy(
            np.ctypeslib.as_array(self.white, shape=(total_ints,))
        )

        # Move the 2 contiguous blocks to the target device in exactly 2 H2D transfers
        float_block_gpu = _pin_and_move(float_block_cpu, device, use_pinned_memory)
        int_block_gpu = _pin_and_move(int_block_cpu, device, use_pinned_memory)

        # Slice the contiguous blocks on the target device (zero-copy operations)
        size = self.size
        max_active = self.max_active_features

        # Slices from float block
        us = float_block_gpu[0 : size].view(size, 1)
        outcome = float_block_gpu[size : 2 * size].view(size, 1)
        score = float_block_gpu[2 * size : 3 * size].view(size, 1)
        
        offset_float = 3 * size
        white_values = float_block_gpu[offset_float : offset_float + size * max_active].view(size, max_active)
        offset_float += size * max_active
        black_values = float_block_gpu[offset_float : offset_float + size * max_active].view(size, max_active)

        # Slices from int block
        white_indices = int_block_gpu[0 : size * max_active].view(size, max_active)
        offset_int = size * max_active
        black_indices = int_block_gpu[offset_int : offset_int + size * max_active].view(size, max_active)
        offset_int += size * max_active
        
        # psqt_indices and layer_stack_indices are sliced and then type-casted to long (int64) on the target device
        psqt_indices = int_block_gpu[offset_int : offset_int + size].view(size).long()
        offset_int += size
        layer_stack_indices = int_block_gpu[offset_int : offset_int + size].view(size).long()

        # Compute 'them' on the target device
        if not us.is_cuda and use_pinned_memory:
            them = torch.empty_like(us, pin_memory=True)
            them.fill_(1.0)
            them.sub_(us)
        else:
            them = 1.0 - us

        return (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        )


class Fen(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("fen", ctypes.c_char_p)]


class FenBatch(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("fens", ctypes.POINTER(Fen))]

    def get_fens(self):
        strings = []
        for i in range(self.size):
            strings.append(self.fens[i].fen.decode("utf-8"))
        return strings


class CDataLoaderAPI:
    def __init__(self):
        self.dll = self._load_library()
        self._define_prototypes()

    def _load_library(self):
        for lib in glob.glob("./build/*training_data_loader.*"):
            if not (
                lib.endswith(".so") or lib.endswith("dll") or lib.endswith(".dylib")
            ):
                continue
            return ctypes.cdll.LoadLibrary(os.path.abspath(lib))
        raise FileNotFoundError("Cannot find data_loader shared library.")

    def _define_prototypes(self):
        # EXPORT FenBatchStream* CDECL create_fen_batch_stream(
        #     int concurrency,
        #     int num_files,
        #     const char* const* filenames,
        #     int batch_size,
        #     bool cyclic,
        #     DataloaderSkipConfig config,
        #     DataloaderDDPConfig ddp_config
        # )
        self.dll.create_fen_batch_stream.restype = ctypes.c_void_p
        self.dll.create_fen_batch_stream.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_bool,
            CDataloaderSkipConfig,
            CDataloaderDDPConfig,
        ]

        # EXPORT void CDECL destroy_fen_batch_stream(FenBatchStream* stream)
        self.dll.destroy_fen_batch_stream.argtypes = [ctypes.c_void_p]

        # EXPORT FenBatch* CDECL fetch_next_fen_batch(Stream<FenBatch>* stream)
        self.dll.fetch_next_fen_batch.restype = ctypes.POINTER(FenBatch)
        self.dll.fetch_next_fen_batch.argtypes = [ctypes.c_void_p]

        # EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(
        #     const char* feature_set_c,
        #     int concurrency,
        #     int num_files,
        #     const char* const* filenames,
        #     int batch_size,
        #     bool cyclic,
        #     DataloaderSkipConfig config,
        #     DataloaderDDPConfig ddp_config
        # )
        self.dll.create_sparse_batch_stream.restype = ctypes.c_void_p
        self.dll.create_sparse_batch_stream.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_bool,
            CDataloaderSkipConfig,
            CDataloaderDDPConfig,
        ]

        # EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
        self.dll.destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

        # EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
        self.dll.fetch_next_sparse_batch.restype = ctypes.POINTER(SparseBatch)
        self.dll.fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]

        # EXPORT SparseBatch* get_sparse_batch_from_fens(
        #    const char* feature_set_c,
        #    int num_fens,
        #    const char* const* fens,
        #    int* scores,
        #    int* plies,
        #    int* results
        # )
        self.dll.get_sparse_batch_from_fens.restype = ctypes.POINTER(SparseBatch)
        self.dll.get_sparse_batch_from_fens.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]


type SparseBatchPtr = ctypes._Pointer[SparseBatch]
type FenBatchPtr = ctypes._Pointer[FenBatch]

try:
    c_lib = CDataLoaderAPI()
except FileNotFoundError as e:
    raise ImportError(f"Failed to initialize CDataLoaderAPI: {e}.")
