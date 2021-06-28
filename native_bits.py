import ctypes
import os
import sys
import glob

local_dllpath = [n for n in glob.glob('./bin/*native_bits.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find native_bits shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)
