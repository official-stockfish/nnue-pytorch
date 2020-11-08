import numpy as np
import os
import sys
import ctypes
import glob
import torch
import dataset_base
import halfkp

local_dllpath = [n for n in glob.glob('./training_data_loader.*') if n.endswith('.so') or n.endswith('.dll')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class DenseEntry(ctypes.Structure):
    _fields_ = [
        ('us', ctypes.c_float),
        ('outcome', ctypes.c_float),
        ('score', ctypes.c_float),
        ('white', ctypes.c_float * halfkp.INPUTS),
        ('black', ctypes.c_float * halfkp.INPUTS)
    ]

    def get_tensors(self):
        us = torch.tensor([self.us])
        them = torch.tensor([1.0 - self.us])
        outcome = torch.tensor([self.outcome])
        score = torch.tensor([self.score])
        white = torch.from_numpy(np.ctypeslib.as_array(self.white)).clone()
        black = torch.from_numpy(np.ctypeslib.as_array(self.black)).clone()
        return us, them, white, black, outcome, score

class SparseEntry(ctypes.Structure):
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
        white = torch.sparse.FloatTensor(iw.unsqueeze(0).long(), torch.ones([self.num_active_white_features], dtype=torch.float32), (halfkp.INPUTS,))
        black = torch.sparse.FloatTensor(ib.unsqueeze(0).long(), torch.ones([self.num_active_black_features], dtype=torch.float32), (halfkp.INPUTS,))
        white.coalesce()
        black.coalesce()
        return us, them, white, black, outcome, score

class DenseBatch(ctypes.Structure):
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
        white = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, halfkp.INPUTS))).clone()
        black = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, halfkp.INPUTS))).clone()
        return us, them, white, black, outcome, score

class SparseBatch(ctypes.Structure):
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
        white = torch.sparse.FloatTensor(iw.long(), torch.ones((self.num_active_white_features), dtype=torch.float32), (self.size, halfkp.INPUTS))
        black = torch.sparse.FloatTensor(ib.long(), torch.ones((self.num_active_black_features), dtype=torch.float32), (self.size, halfkp.INPUTS))
        return us, them, white, black, outcome, score

DenseEntryPtr = ctypes.POINTER(DenseEntry)
SparseEntryPtr = ctypes.POINTER(SparseEntry)
SparseBatchPtr = ctypes.POINTER(SparseBatch)
DenseBatchPtr = ctypes.POINTER(DenseBatch)

create_halfkp_dense_entry_stream = dll.create_halfkp_dense_entry_stream
destroy_halfkp_dense_entry_stream = dll.destroy_halfkp_dense_entry_stream

create_halfkp_sparse_entry_stream = dll.create_halfkp_sparse_entry_stream
destroy_halfkp_sparse_entry_stream = dll.destroy_halfkp_sparse_entry_stream

create_halfkp_dense_batch_stream = dll.create_halfkp_dense_batch_stream
destroy_halfkp_dense_batch_stream = dll.destroy_halfkp_dense_batch_stream

create_halfkp_sparse_batch_stream = dll.create_halfkp_sparse_batch_stream
destroy_halfkp_sparse_batch_stream = dll.destroy_halfkp_sparse_batch_stream

get_next_halfkp_dense_entry = dll.get_next_halfkp_dense_entry
get_next_halfkp_dense_entry.restype = DenseEntryPtr
destroy_halfkp_dense_entry = dll.destroy_halfkp_dense_entry

get_next_halfkp_sparse_entry = dll.get_next_halfkp_sparse_entry
get_next_halfkp_sparse_entry.restype = SparseEntryPtr
destroy_halfkp_sparse_entry = dll.destroy_halfkp_sparse_entry

get_next_halfkp_dense_batch = dll.get_next_halfkp_dense_batch
get_next_halfkp_dense_batch.restype = DenseBatchPtr
destroy_halfkp_dense_batch = dll.destroy_halfkp_dense_batch

get_next_halfkp_sparse_batch = dll.get_next_halfkp_sparse_batch
get_next_halfkp_sparse_batch.restype = SparseBatchPtr
destroy_halfkp_sparse_batch = dll.destroy_halfkp_sparse_batch

class DenseEntryProvider(dataset_base.TrainingDataProvider):
    def __init__(self, filename):
        super(DenseEntryProvider, self).__init__(
            create_halfkp_dense_entry_stream,
            destroy_halfkp_dense_entry_stream,
            get_next_halfkp_dense_entry,
            destroy_halfkp_dense_entry,
            filename)

class SparseEntryProvider(dataset_base.TrainingDataProvider):
    def __init__(self, filename):
        super(SparseEntryProvider, self).__init__(
            create_halfkp_sparse_entry_stream,
            destroy_halfkp_sparse_entry_stream,
            get_next_halfkp_sparse_entry,
            destroy_halfkp_sparse_entry,
            filename)

class DenseBatchProvider(dataset_base.TrainingDataProvider):
    def __init__(self, filename, batch_size):
        super(DenseBatchProvider, self).__init__(
            create_halfkp_dense_batch_stream,
            destroy_halfkp_dense_batch_stream,
            get_next_halfkp_dense_batch,
            destroy_halfkp_dense_batch,
            filename,
            batch_size)

class SparseBatchProvider(dataset_base.TrainingDataProvider):
    def __init__(self, filename, batch_size):
        super(SparseBatchProvider, self).__init__(
            create_halfkp_sparse_batch_stream,
            destroy_halfkp_sparse_batch_stream,
            get_next_halfkp_sparse_batch,
            destroy_halfkp_sparse_batch,
            filename,
            batch_size)

class DenseEntryDataset(torch.utils.data.IterableDataset):
  def __init__(self, filename):
    super(DenseEntryDataset).__init__()
    self.filename = filename

  def __iter__(self):
    return DenseEntryProvider(self.filename)

class SparseEntryDataset(torch.utils.data.IterableDataset):
  def __init__(self, filename):
    super(SparseEntryDataset).__init__()
    self.filename = filename

  def __iter__(self):
    return SparseEntryProvider(self.filename)

class DenseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, filename, batch_size):
    super(DenseBatchDataset).__init__()
    self.filename = filename
    self.batch_size = batch_size

  def __iter__(self):
    return DenseBatchProvider(self.filename, self.batch_size)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, filename, batch_size):
    super(SparseBatchDataset).__init__()
    self.filename = filename
    self.batch_size = batch_size

  def __iter__(self):
    return SparseBatchProvider(self.filename, self.batch_size)