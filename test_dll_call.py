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

for i in range(1000000):
    v = create_data_collection()
    destroy_data_collection(v)