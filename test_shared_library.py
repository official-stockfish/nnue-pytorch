import nnue_dataset
import halfkp
import time

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

test_stream(nnue_dataset.DenseEntryProvider)
test_stream(nnue_dataset.SparseEntryProvider)
test_stream(nnue_dataset.DenseBatchProvider, 64)
test_stream(nnue_dataset.SparseBatchProvider, 256)