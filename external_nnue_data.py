import numpy as np
import os
import ctypes

dll = ctypes.CDLL('c:/dev/nnue-pytorch/data_loader.dll')

import torch

class TrainingEntryHalfKPDense(ctypes.Structure):
    _fields_ = [
        ('us', ctypes.c_float),
        ('outcome', ctypes.c_float),
        ('score', ctypes.c_float),
        ('white', ctypes.c_float * 41024),
        ('black', ctypes.c_float * 41024)
    ]

    def get_tensors(self):
        us = torch.tensor([self.us])
        them = torch.tensor([1.0 - self.us])
        outcome = torch.tensor([self.outcome])
        score = torch.tensor([self.score])
        white = torch.from_numpy(np.ctypeslib.as_array(self.white))
        black = torch.from_numpy(np.ctypeslib.as_array(self.black))
        return us, them, white, black, outcome, score

class TrainingEntryHalfKPSparse(ctypes.Structure):
    _fields_ = [
        ('us', ctypes.c_float),
        ('outcome', ctypes.c_float),
        ('score', ctypes.c_float),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('white', ctypes.c_int * 32),
        ('black', ctypes.c_int * 32)
    ]

    def get_tensors(self):
        us = torch.tensor([self.us])
        them = torch.tensor([1.0 - self.us])
        outcome = torch.tensor([self.outcome])
        score = torch.tensor([self.score])
        white = torch.empty(41024, layout=torch.sparse_coo)
        black = torch.empty(41024, layout=torch.sparse_coo)
        for i in range(self.num_active_white_features):
            white[self.white[i]] = 1.0
        for i in range(self.num_active_black_features):
            black[self.black[i]] = 1.0
        white.coalesce()
        black.coalesce()
        return us, them, white, black, outcome, score

TrainingEntryHalfKPDensePtr = ctypes.POINTER(TrainingEntryHalfKPDense)
TrainingEntryHalfKPSparsePtr = ctypes.POINTER(TrainingEntryHalfKPSparse)

create_stream = dll.create_stream
destroy_stream = dll.destroy_stream
print_next = dll.print_next
get_next_entry_halfkp_dense = dll.get_next_entry_halfkp_dense
get_next_entry_halfkp_dense.restype = TrainingEntryHalfKPDensePtr
destroy_entry_halfkp_dense = dll.destroy_entry_halfkp_dense
get_next_entry_halfkp_sparse = dll.get_next_entry_halfkp_sparse
get_next_entry_halfkp_sparse.restype = TrainingEntryHalfKPSparsePtr
destroy_entry_halfkp_sparse = dll.destroy_entry_halfkp_sparse


class NNUEExternalDataDenseIterator:
    def __init__(self, filename):
        self.filename = filename
        self.stream = create_stream(filename.encode('utf-8'))

    def __iter__(self):
        return self

    def __next__(self):
        v = get_next_entry_halfkp_dense(self.stream)
        if v:
            tensors = v.contents.get_tensors()
            destroy_entry_halfkp_dense(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        destroy_stream(self.stream)

class NNUEExternalDataSparseIterator:
    def __init__(self, filename):
        self.filename = filename
        self.stream = create_stream(filename.encode('utf-8'))

    def __iter__(self):
        return self

    def __next__(self):
        v = get_next_entry_halfkp_sparse(self.stream)
        if v:
            tensors = v.contents.get_tensors()
            destroy_entry_halfkp_sparse(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        destroy_stream(self.stream)