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
        white = torch.from_numpy(np.ctypeslib.as_array(self.white)).clone()
        black = torch.from_numpy(np.ctypeslib.as_array(self.black)).clone()
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
        iw = torch.from_numpy(np.ctypeslib.as_array(self.white)[:self.num_active_white_features])
        ib = torch.from_numpy(np.ctypeslib.as_array(self.black)[:self.num_active_black_features])
        white = torch.sparse.FloatTensor(iw.unsqueeze(0).long(), torch.ones([self.num_active_white_features], dtype=torch.float32), (41024,))
        black = torch.sparse.FloatTensor(ib.unsqueeze(0).long(), torch.ones([self.num_active_black_features], dtype=torch.float32), (41024,))
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

class TrainingEntryHalfKPSparseBatch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('us', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int))
    ]

    def get_tensors(self):
        us = torch.from_numpy(np.ctypeslib.as_array(self.us, shape=(self.size, 1))).clone()
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).clone()
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).clone()
        iw = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.num_active_white_features, 2)).transpose()).clone()
        ib = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.num_active_white_features, 2)).transpose()).clone()
        white = torch.sparse.FloatTensor(iw.long(), torch.ones((self.num_active_white_features), dtype=torch.float32), (self.size, 41024))
        black = torch.sparse.FloatTensor(ib.long(), torch.ones((self.num_active_black_features), dtype=torch.float32), (self.size, 41024))
        white.coalesce()
        black.coalesce()
        return us, them, white, black, outcome, score

TrainingEntryHalfKPSparseBatchPtr = ctypes.POINTER(TrainingEntryHalfKPSparseBatch)

get_next_entry_halfkp_sparse_batch = dll.get_next_entry_halfkp_sparse_batch
get_next_entry_halfkp_sparse_batch.restype = TrainingEntryHalfKPSparseBatchPtr
destroy_entry_halfkp_sparse_batch = dll.destroy_entry_halfkp_sparse_batch

class TrainingEntryHalfKPDenseBatch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('us', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('white', ctypes.POINTER(ctypes.c_float)),
        ('black', ctypes.POINTER(ctypes.c_float))
    ]

    def get_tensors(self):
        us = torch.from_numpy(np.ctypeslib.as_array(self.us, shape=(self.size, 1))).clone()
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).clone()
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).clone()
        white = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, 41024))).clone()
        black = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, 41024))).clone()
        return us, them, white, black, outcome, score

TrainingEntryHalfKPDenseBatchPtr = ctypes.POINTER(TrainingEntryHalfKPDenseBatch)

get_next_entry_halfkp_dense_batch = dll.get_next_entry_halfkp_dense_batch
get_next_entry_halfkp_dense_batch.restype = TrainingEntryHalfKPDenseBatchPtr
destroy_entry_halfkp_dense_batch = dll.destroy_entry_halfkp_dense_batch

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

class NNUEExternalDataDenseBatchIterator:
    def __init__(self, filename, batch_size):
        self.filename = filename
        self.stream = create_stream(filename.encode('utf-8'))
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        v = get_next_entry_halfkp_dense_batch(self.stream, self.batch_size)
        if v.contents.size:
            tensors = v.contents.get_tensors()
            destroy_entry_halfkp_dense_batch(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        destroy_stream(self.stream)

class NNUEExternalDataSparseBatchIterator:
    def __init__(self, filename, batch_size):
        self.filename = filename
        self.stream = create_stream(filename.encode('utf-8'))
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        v = get_next_entry_halfkp_sparse_batch(self.stream, self.batch_size)
        if v.contents.size:
            tensors = v.contents.get_tensors()
            destroy_entry_halfkp_sparse_batch(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        destroy_stream(self.stream)