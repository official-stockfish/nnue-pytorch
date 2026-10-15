import ctypes
import glob
import os

import numpy as np
import torch

from ._transport import CompactTransportPolicy, _pin_and_move, move_indices
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
        ("white", ctypes.POINTER(ctypes.c_int)),
        ("black", ctypes.POINTER(ctypes.c_int)),
        ("piece_count", ctypes.POINTER(ctypes.c_int)),
        ("queen_bucket", ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(
        self,
        device: str | torch.device,
        use_pinned_memory: bool = False,
        *,
        transport_policy: CompactTransportPolicy | None = None,
    ) -> tuple[
        torch.Tensor,
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

        # is_white, outcome, score(3 * size floats)
        total_floats = size * 3

        float_block_cpu = torch.from_numpy(
            np.ctypeslib.as_array(self.is_white, shape=(total_floats,))
        )
        float_block_gpu = _pin_and_move(float_block_cpu, device, use_pinned_memory)
        int_block_gpu = move_indices(
            self,
            device,
            use_pinned_memory,
            native_owned=isinstance(self._b_base_, NativeSparseBatchPointer),
            pack=c_lib.pack_sparse_batch,
            policy=transport_policy or CompactTransportPolicy(),
        )

        us = float_block_gpu[0:size].view(size, 1)
        outcome = float_block_gpu[size : 2 * size].view(size, 1)
        score = float_block_gpu[2 * size : 3 * size].view(size, 1)

        white_indices = int_block_gpu[0 : size * max_active].view(size, max_active)
        black_indices = int_block_gpu[size * max_active : 2 * size * max_active].view(
            size, max_active
        )
        piece_count_i32 = int_block_gpu[
            2 * size * max_active : 2 * size * max_active + size
        ].view(size)
        queen_bucket_i32 = int_block_gpu[
            2 * size * max_active + size : 2 * size * max_active + 2 * size
        ].view(size)

        # Keep piece counts and queen buckets as int64 so callers can derive buckets on the target device.
        if not us.is_cuda and use_pinned_memory:
            them = torch.empty_like(us, pin_memory=True)
            them.fill_(1.0)
            them.sub_(us)

            piece_count = torch.empty(
                size, dtype=torch.int64, device="cpu", pin_memory=True
            )
            piece_count.copy_(piece_count_i32)

            queen_bucket = torch.empty(
                size, dtype=torch.int64, device="cpu", pin_memory=True
            )
            queen_bucket.copy_(queen_bucket_i32)
        else:
            them = 1.0 - us
            piece_count = piece_count_i32.to(dtype=torch.int64)
            queen_bucket = queen_bucket_i32.to(dtype=torch.int64)

        return (
            us,
            them,
            white_indices,
            black_indices,
            outcome,
            score,
            piece_count,
            queen_bucket,
        )


class NativeSparseBatchPointer(ctypes._Pointer):
    """Distinguish native allocations with private fields from Python-created batches."""

    _type_ = SparseBatch


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
        self.pack_sparse_batch = self.dll.pack_sparse_batch
        self.pack_sparse_batch.restype = ctypes.c_bool
        self.pack_sparse_batch.argtypes = [
            ctypes.POINTER(SparseBatch),
            ctypes.c_void_p,
            ctypes.c_size_t,
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
        self.dll.fetch_next_sparse_batch.restype = NativeSparseBatchPointer
        self.dll.fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]

        # EXPORT SparseBatch* get_sparse_batch_from_fens(
        #    const char* feature_set_c,
        #    int num_fens,
        #    const char* const* fens,
        #    int* scores,
        #    int* plies,
        #    int* results
        # )
        self.dll.get_sparse_batch_from_fens.restype = NativeSparseBatchPointer
        self.dll.get_sparse_batch_from_fens.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
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


type SparseBatchPtr = NativeSparseBatchPointer
type FenBatchPtr = ctypes._Pointer[FenBatch]

try:
    c_lib = CDataLoaderAPI()
except FileNotFoundError as e:
    raise ImportError(f"Failed to initialize CDataLoaderAPI: {e}.")
