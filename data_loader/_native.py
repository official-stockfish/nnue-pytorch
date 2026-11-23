import ctypes
import glob
import os

import numpy as np
import torch

from ._transport import copy_to_device, move_indices
from .config import (
    CDataloaderDDPConfig,
    CDataloaderFileStats,
    CDataloaderHllConfig,
    CDataloaderIOConfig,
    CDataloaderSkipConfig,
)


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
        # Feature IDs, packed: white offsets [size + 1], black offsets [size + 1], white IDs,
        # black IDs, layer-stack indices [size].
        ("packed", ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(
        self,
        device: str | torch.device,
        use_pinned_memory: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        size = self.size
        max_active = self.max_active_features

        # is_white, outcome, score (3 * size floats)
        float_block_cpu = torch.from_numpy(
            np.ctypeslib.as_array(self.is_white, shape=(size * 3,))
        )
        float_block_gpu = copy_to_device(float_block_cpu, device, use_pinned_memory)
        int_block_gpu = move_indices(
            self, device, use_pinned_memory, unpack=c_lib.unpack_sparse_batch
        )

        us = float_block_gpu[0:size].view(size, 1)
        outcome = float_block_gpu[size : 2 * size].view(size, 1)
        score = float_block_gpu[2 * size : 3 * size].view(size, 1)

        white_indices = int_block_gpu[0 : size * max_active].view(size, max_active)
        black_indices = int_block_gpu[size * max_active : 2 * size * max_active].view(
            size, max_active
        )
        layer_stack_index_i32 = int_block_gpu[
            2 * size * max_active : 2 * size * max_active + size
        ].view(size)

        # Keep layer-stack indices as int64 for indexing on the target device.
        if not us.is_cuda and use_pinned_memory:
            them = torch.empty_like(us, pin_memory=True)
            them.fill_(1.0)
            them.sub_(us)

            layer_stack_index = torch.empty(
                size, dtype=torch.int64, device="cpu", pin_memory=True
            )
            layer_stack_index.copy_(layer_stack_index_i32)
        else:
            them = 1.0 - us
            layer_stack_index = layer_stack_index_i32.to(dtype=torch.int64)

        return (
            us,
            them,
            white_indices,
            black_indices,
            outcome,
            score,
            layer_stack_index,
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
        candidates = []
        for pattern in (
            "./build/**/*training_data_loader.*",
            "./build/*training_data_loader.*",
            os.path.join(
                os.path.dirname(__file__),
                "cpp",
                "build",
                "**",
                "*training_data_loader.*",
            ),
            os.path.join(
                os.path.dirname(__file__), "cpp", "build", "*training_data_loader.*"
            ),
        ):
            for lib in glob.glob(pattern, recursive=True):
                if lib.endswith((".so", ".dll", ".dylib")):
                    candidates.append(os.path.abspath(lib))

        last_error: OSError | None = None
        for lib in sorted(set(candidates), key=os.path.getmtime, reverse=True):
            try:
                return ctypes.cdll.LoadLibrary(lib)
            except OSError as e:
                last_error = e

        if last_error is not None:
            raise OSError(
                "Found data_loader shared libraries but failed to load any of them."
            ) from last_error
        raise FileNotFoundError("Cannot find data_loader shared library.")

    def _define_prototypes(self):
        # EXPORT void CDECL unpack_sparse_batch(const SparseBatch* batch, int* dense)
        self.unpack_sparse_batch = self.dll.unpack_sparse_batch
        self.unpack_sparse_batch.restype = None
        self.unpack_sparse_batch.argtypes = [
            ctypes.POINTER(SparseBatch),
            ctypes.c_void_p,
        ]
        # EXPORT FenBatchStream* CDECL create_fen_batch_stream(
        #     int concurrency,
        #     int num_files,
        #     const char* const* filenames,
        #     int batch_size,
        #     bool cyclic,
        #     DataloaderSkipConfig config,
        #     DataloaderDDPConfig ddp_config,
        #     DataloaderHllConfig hll_config,
        #     DataloaderIOConfig io_config
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
            CDataloaderHllConfig,
            CDataloaderIOConfig,
        ]

        # EXPORT void CDECL destroy_fen_batch_stream(FenBatchStream* stream)
        self.dll.destroy_fen_batch_stream.argtypes = [ctypes.c_void_p]

        # EXPORT FenBatch* CDECL fetch_next_fen_batch(Stream<FenBatch>* stream)
        self.dll.fetch_next_fen_batch.restype = ctypes.POINTER(FenBatch)
        self.dll.fetch_next_fen_batch.argtypes = [ctypes.c_void_p]

        # EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(
        #     const char* feature_set_c,
        #     const char* ls_keys_c,
        #     int concurrency,
        #     int num_files,
        #     const char* const* filenames,
        #     int batch_size,
        #     bool cyclic,
        #     DataloaderSkipConfig config,
        #     DataloaderDDPConfig ddp_config,
        #     DataloaderHllConfig hll_config,
        #     DataloaderIOConfig io_config
        # )
        self.dll.create_sparse_batch_stream.restype = ctypes.c_void_p
        self.dll.create_sparse_batch_stream.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_bool,
            CDataloaderSkipConfig,
            CDataloaderDDPConfig,
            CDataloaderHllConfig,
            CDataloaderIOConfig,
        ]

        # EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
        self.dll.destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

        # EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
        self.dll.fetch_next_sparse_batch.restype = ctypes.POINTER(SparseBatch)
        self.dll.fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]

        # EXPORT SparseBatch* get_sparse_batch_from_fens(
        #    const char* feature_set_c,
        #    const char* ls_keys_c,
        #    int num_fens,
        #    const char* const* fens,
        #    int* scores,
        #    int* plies,
        #    int* results
        # )
        self.dll.get_sparse_batch_from_fens.restype = ctypes.POINTER(SparseBatch)
        self.dll.get_sparse_batch_from_fens.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        # EXPORT bool CDECL count_layer_stacks(
        #     const char* ls_keys_c,
        #     int concurrency,
        #     int num_files,
        #     const char* const* filenames,
        #     DataloaderDDPConfig ddp_config,
        #     uint64_t* out_counts,
        #     int out_size
        # )
        self.dll.count_layer_stacks.restype = ctypes.c_bool
        self.dll.count_layer_stacks.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            CDataloaderDDPConfig,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
        ]

        # EXPORT void CDECL get_unique_position_stats(SparseBatchStream*, uint64_t*, uint64_t*, uint64_t*)
        self.dll.get_unique_position_stats.restype = None
        self.dll.get_unique_position_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]

        # EXPORT size_t CDECL get_hll_state(SparseBatchStream*, uint8_t*, size_t)
        self.dll.get_hll_state.restype = ctypes.c_size_t
        self.dll.get_hll_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]

        # EXPORT size_t CDECL get_hll_state_size(SparseBatchStream*)
        self.dll.get_hll_state_size.restype = ctypes.c_size_t
        self.dll.get_hll_state_size.argtypes = [ctypes.c_void_p]

        # EXPORT void CDECL hll_count_from_state(const uint8_t*, size_t, uint64_t*)
        self.dll.hll_count_from_state.restype = None
        self.dll.hll_count_from_state.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]

        # EXPORT size_t CDECL get_io_stats(SparseBatchStream*, DataloaderFileStats*, size_t)
        self.dll.get_io_stats.restype = ctypes.c_size_t
        self.dll.get_io_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CDataloaderFileStats),
            ctypes.c_size_t,
        ]


type SparseBatchPtr = ctypes._Pointer[SparseBatch]
type FenBatchPtr = ctypes._Pointer[FenBatch]

try:
    c_lib = CDataLoaderAPI()
except FileNotFoundError as e:
    raise ImportError(f"Failed to initialize CDataLoaderAPI: {e}.")
