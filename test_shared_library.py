import nnue_dataset
import halfkp
import time
import torch
import nnue_bin_dataset

def test_stream(stream_type, batch_size=None):
    if batch_size:
        stream = stream_type(halfkp.NAME, 'd8_100000.bin', batch_size)
    else:
        stream = stream_type(halfkp.NAME, 'd8_100000.bin')

    start_time = time.time()
    for i in range(4096 // (batch_size if batch_size else 1)):
        tensors = next(stream)
    end_time = time.time()
    print('{:6.3f} seconds'.format(end_time-start_time))

    del stream

test_stream(nnue_dataset.SparseBatchProvider, 256)

stream_py = nnue_bin_dataset.NNUEBinData('d8_100000.bin')
stream_cpp = nnue_dataset.SparseBatchDataset(halfkp.NAME, 'd8_100000.bin', 256)

stream_py_iter = iter(stream_py)
stream_cpp_iter = iter(stream_cpp)

diff = 0.0
for i in range(10):
    # Gather a batch
    tensors_cpp = next(stream_cpp_iter)
    for j in range(256):
        tensors_py = next(stream_py_iter)
        diff += sum((a - b[j]).norm() for a, b in zip(tensors_py, tensors_cpp))
print('Diff: {}'.format(diff))

