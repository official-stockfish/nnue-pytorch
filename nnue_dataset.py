import numpy as np
import ctypes
import torch
import os
import sys
import glob

local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class DenseEntry(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('is_white', ctypes.c_float),
        ('outcome', ctypes.c_float),
        ('score', ctypes.c_float),
        ('white', ctypes.POINTER(ctypes.c_float)),
        ('black', ctypes.POINTER(ctypes.c_float))
    ]

    def get_tensors(self):
        us = torch.tensor([self.is_white])
        them = torch.tensor([1.0 - self.is_white])
        outcome = torch.tensor([self.outcome])
        score = torch.tensor([self.score])
        white = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.num_inputs, ))).clone()
        black = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.num_inputs, ))).clone()
        return us, them, white, black, outcome, score

class SparseEntry(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('is_white', ctypes.c_float),
        ('outcome', ctypes.c_float),
        ('score', ctypes.c_float),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int))
    ]

    def get_tensors(self):
        us = torch.tensor([self.is_white])
        them = torch.tensor([1.0 - self.is_white])
        outcome = torch.tensor([self.outcome])
        score = torch.tensor([self.score])
        iw = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.num_active_white_features,)))
        ib = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.num_active_black_features,)))
        white = torch.sparse.FloatTensor(iw.unsqueeze(0).long(), torch.ones([self.num_active_white_features], dtype=torch.float32), (self.num_inputs,))
        black = torch.sparse.FloatTensor(ib.unsqueeze(0).long(), torch.ones([self.num_active_black_features], dtype=torch.float32), (self.num_inputs,))
        white.coalesce()
        black.coalesce()
        return us, them, white, black, outcome, score

class DenseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('white', ctypes.POINTER(ctypes.c_float)),
        ('black', ctypes.POINTER(ctypes.c_float))
    ]

    def get_tensors(self):
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).clone()
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).clone()
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).clone()
        white = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.num_inputs))).clone()
        black = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.num_inputs))).clone()
        return us, them, white, black, outcome, score

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int))
    ]

    def get_tensors(self):
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).clone()
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).clone()
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).clone()
        iw = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.num_active_white_features, 2)).transpose()).clone()
        ib = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.num_active_white_features, 2)).transpose()).clone()
        white = torch.sparse.FloatTensor(iw.long(), torch.ones((self.num_active_white_features), dtype=torch.float32), (self.size, self.num_inputs))
        black = torch.sparse.FloatTensor(ib.long(), torch.ones((self.num_active_black_features), dtype=torch.float32), (self.size, self.num_inputs))
        return us, them, white, black, outcome, score

DenseEntryPtr = ctypes.POINTER(DenseEntry)
SparseEntryPtr = ctypes.POINTER(SparseEntry)
SparseBatchPtr = ctypes.POINTER(SparseBatch)
DenseBatchPtr = ctypes.POINTER(DenseBatch)

class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename,
        cyclic,
        batch_size=None):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.batch_size = batch_size

        if batch_size:
            self.stream = self.create_stream(self.feature_set, self.filename, batch_size, cyclic)
        else:
            self.stream = self.create_stream(self.feature_set, self.filename, cyclic)

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors()
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)

create_dense_entry_stream = dll.create_dense_entry_stream
destroy_dense_entry_stream = dll.destroy_dense_entry_stream

create_sparse_entry_stream = dll.create_sparse_entry_stream
destroy_sparse_entry_stream = dll.destroy_sparse_entry_stream

create_dense_batch_stream = dll.create_dense_batch_stream
destroy_dense_batch_stream = dll.destroy_dense_batch_stream

create_sparse_batch_stream = dll.create_sparse_batch_stream
destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream

fetch_next_dense_entry = dll.fetch_next_dense_entry
fetch_next_dense_entry.restype = DenseEntryPtr
destroy_dense_entry = dll.destroy_dense_entry

fetch_next_sparse_entry = dll.fetch_next_sparse_entry
fetch_next_sparse_entry.restype = SparseEntryPtr
destroy_sparse_entry = dll.destroy_sparse_entry

fetch_next_dense_batch = dll.fetch_next_dense_batch
fetch_next_dense_batch.restype = DenseBatchPtr
destroy_dense_batch = dll.destroy_dense_batch

fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
destroy_sparse_batch = dll.destroy_sparse_batch

class DenseEntryProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, cyclic=True):
        super(DenseEntryProvider, self).__init__(
            feature_set,
            create_dense_entry_stream,
            destroy_dense_entry_stream,
            fetch_next_dense_entry,
            destroy_dense_entry,
            filename,
            cyclic)

class SparseEntryProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, cyclic=True):
        super(SparseEntryProvider, self).__init__(
            feature_set,
            create_sparse_entry_stream,
            destroy_sparse_entry_stream,
            fetch_next_sparse_entry,
            destroy_sparse_entry,
            filename,
            cyclic)

class DenseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True):
        super(DenseBatchProvider, self).__init__(
            feature_set,
            create_dense_batch_stream,
            destroy_dense_batch_stream,
            fetch_next_dense_batch,
            destroy_dense_batch,
            filename,
            cyclic,
            batch_size)

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename,
            cyclic,
            batch_size)

class DenseEntryDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, cyclic=True):
    super(DenseEntryDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.cyclic = cyclic

  def __iter__(self):
    return DenseEntryProvider(self.feature_set, self.filename, cyclic=self.cyclic)

class SparseEntryDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, cyclic=True):
    super(SparseEntryDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.cyclic = cyclic

  def __iter__(self):
    return SparseEntryProvider(self.feature_set, self.filename, cyclic=self.cyclic)

class DenseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True):
    super(DenseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic

  def __iter__(self):
    return DenseBatchProvider(self.feature_set, self.filename, self.batch_size, cyclic=self.cyclic)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filename, self.batch_size, cyclic=self.cyclic)
