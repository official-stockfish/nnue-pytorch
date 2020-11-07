import ctypes
import numpy as np

dll = ctypes.CDLL('./data_loader.dll')
print(dll)
print(dll.test)
dll.test()

import torch

print(dll.create_data_collection)
print(dll.destroy_data_collection)

class TestDataCollection(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('data', ctypes.POINTER(ctypes.c_int))
    ]

    def __str__(self):
        return str(self.size) + ' ' + str([self.data[i] for i in range(self.size)])

    def get_tensor(self):
        return torch.LongTensor(np.ctypeslib.as_array(self.data, shape=(self.size,)))

TestDataCollectionPtr = ctypes.POINTER(TestDataCollection)

create_data_collection = dll.create_data_collection
create_data_collection.restype = TestDataCollectionPtr

destroy_data_collection = dll.destroy_data_collection
destroy_data_collection.argtypes = [TestDataCollectionPtr]

v = dll.create_data_collection()
print(v)
vobj = v.contents
print(vobj)

tensor = vobj.get_tensor();
print(tensor)

destroy_data_collection(v)

'''
for i in range(1000000):
    v = create_data_collection()
    destroy_data_collection(v)
'''

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
        white = torch.tensor(np.ctypeslib.as_array(self.white))
        black = torch.tensor(np.ctypeslib.as_array(self.black))
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

stream = create_stream(b'd10_10000.bin')
print(stream)
for i in range(100):
    e = get_next_entry_halfkp_sparse(stream)
    e.contents.get_tensors()
    destroy_entry_halfkp_sparse(e)

destroy_stream(stream)