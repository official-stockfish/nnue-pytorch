import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset

local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float)),
        ('psqt_indices', ctypes.POINTER(ctypes.c_int)),
        ('layer_stack_indices', ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device):
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        white_indices = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_indices = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        psqt_indices = torch.from_numpy(np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        layer_stack_indices = torch.from_numpy(np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        return us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices

SparseBatchPtr = ctypes.POINTER(SparseBatch)

class Fen(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('fen', ctypes.c_char_p)
    ]

FenPtr = ctypes.POINTER(Fen)

class FenBatch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('fens', FenPtr)
    ]

    def get_fens(self):
        strings = []
        for i in range(self.size):
            strings.append(self.fens[i].fen.decode('utf-8'))
        return strings

FenBatchPtr = ctypes.POINTER(FenBatch)
# EXPORT FenBatchStream* CDECL create_fen_batch_stream(int concurrency, int num_files, const char* const* filenames, int batch_size, bool cyclic, bool filtered, int random_fen_skipping, bool wld_filtered, int early_fen_skipping, int param_index)
create_fen_batch_stream = dll.create_fen_batch_stream
create_fen_batch_stream.restype = ctypes.c_void_p
create_fen_batch_stream.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
destroy_fen_batch_stream = dll.destroy_fen_batch_stream
destroy_fen_batch_stream.argtypes = [ctypes.c_void_p]

def make_fen_batch_stream(concurrency, filenames, batch_size, cyclic, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index):
    filenames_ = (ctypes.c_char_p * len(filenames))()
    filenames_[:] = [filename.encode('utf-8') for filename in filenames]
    return create_fen_batch_stream(concurrency, len(filenames), filenames_, batch_size, cyclic, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index)

fetch_next_fen_batch = dll.fetch_next_fen_batch
fetch_next_fen_batch.restype = FenBatchPtr
fetch_next_fen_batch.argtypes = [ctypes.c_void_p]
destroy_fen_batch = dll.destroy_fen_batch

class FenBatchProvider:
    def __init__(
        self,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        early_fen_skipping=-1,
        wld_filtered=False,
        param_index=0):

        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.wld_filtered = wld_filtered
        self.random_fen_skipping = random_fen_skipping
        self.early_fen_skipping = early_fen_skipping
        self.param_index = param_index

        if batch_size:
            self.stream = make_fen_batch_stream(self.num_workers, [self.filename], batch_size, cyclic, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index)
        else:
            self.stream = make_fen_batch_stream(self.num_workers, [self.filename], cyclic, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index)

    def __iter__(self):
        return self

    def __next__(self):
        v = fetch_next_fen_batch(self.stream)

        if v:
            fens = v.contents.get_fens()
            destroy_fen_batch(v)
            return fens
        else:
            raise StopIteration

    def __del__(self):
        destroy_fen_batch_stream(self.stream)

class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filenames,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        wld_filtered=False,
        early_fen_skipping=-1,
        param_index=0,
        device='cpu'):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filenames = filenames
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.wld_filtered = wld_filtered
        self.random_fen_skipping = random_fen_skipping
        self.param_index = param_index
        self.device = device

        if batch_size:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filenames, batch_size, cyclic, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index)
        else:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filenames, cyclic, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index)

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors(self.device)
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)

#    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, int num_files, const char* const* filenames, int batch_size, bool cyclic,
#                                                                 bool filtered, int random_fen_skipping, bool wld_filtered, int early_fen_skipping, int param_index)
create_sparse_batch_stream = dll.create_sparse_batch_stream
create_sparse_batch_stream.restype = ctypes.c_void_p
create_sparse_batch_stream.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

def make_sparse_batch_stream(feature_set, concurrency, filenames, batch_size, cyclic, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index):
    filenames_ = (ctypes.c_char_p * len(filenames))()
    filenames_[:] = [filename.encode('utf-8') for filename in filenames]
    return create_sparse_batch_stream(feature_set, concurrency, len(filenames), filenames_, batch_size, cyclic, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index)

fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
destroy_sparse_batch = dll.destroy_sparse_batch

get_sparse_batch_from_fens = dll.get_sparse_batch_from_fens
get_sparse_batch_from_fens.restype = SparseBatchPtr
get_sparse_batch_from_fens.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

def make_sparse_batch_from_fens(feature_set, fens, scores, plies, results):
    results_ = (ctypes.c_int*len(scores))()
    scores_ = (ctypes.c_int*len(plies))()
    plies_ = (ctypes.c_int*len(results))()
    fens_ = (ctypes.c_char_p * len(fens))()
    fens_[:] = [fen.encode('utf-8') for fen in fens]
    for i, v in enumerate(scores):
        scores_[i] = v
    for i, v in enumerate(plies):
        plies_[i] = v
    for i, v in enumerate(results):
        results_[i] = v
    b = get_sparse_batch_from_fens(feature_set.name.encode('utf-8'), len(fens), fens_, scores_, plies_, results_)
    return b

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filenames, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, wld_filtered=False, early_fen_skipping=-1, param_index=0, device='cpu'):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            make_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filenames,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            wld_filtered,
            early_fen_skipping,
            param_index,
            device)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filenames, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, wld_filtered=False, early_fen_skipping=-1, param_index=0, device='cpu'):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filenames = filenames
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.filtered = filtered
    self.random_fen_skipping = random_fen_skipping
    self.wld_filtered = wld_filtered
    self.early_fen_skipping = early_fen_skipping
    self.param_index = param_index
    self.device = device

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filenames, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers,
                               filtered=self.filtered, random_fen_skipping=self.random_fen_skipping, wld_filtered=self.wld_filtered, early_fen_skipping = self.early_fen_skipping, param_index=self.param_index, device=self.device)

class FixedNumBatchesDataset(Dataset):
  def __init__(self, dataset, num_batches):
    super(FixedNumBatchesDataset, self).__init__()
    self.dataset = dataset;
    self.iter = iter(self.dataset)
    self.num_batches = num_batches

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return next(self.iter)
