from ctypes import *
from ctypes.util import find_library

filename = 'data_loader.dll'
dll = CDLL(filename)
print(dll)
print(dll.test)
dll.test()


