import ctypes
import os
import glob

import numpy as np
import torch

from .config import CDataloaderSkipConfig, CDataloaderDDPConfig


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

    def get_tensors(self, device):
        white_values = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.white_values, shape=(self.size, self.max_active_features)
                )
            )
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        black_values = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.black_values, shape=(self.size, self.max_active_features)
                )
            )
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        white_indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.white, shape=(self.size, self.max_active_features)
                )
            )
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        black_indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.black, shape=(self.size, self.max_active_features)
                )
            )
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        us = (
            torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1)))
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        them = 1.0 - us
        outcome = (
            torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1)))
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        score = (
            torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1)))
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        psqt_indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))
            )
            .long()
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
        layer_stack_indices = (
            torch.from_numpy(
                np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))
            )
            .long()
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
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
        for lib in glob.glob("./*training_data_loader.*"):
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
    print(e)
    exit(1)
