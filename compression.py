import ctypes
import numpy as np

from native_bits import dll

'''
A class representing a block of compressed data that can be decompressed using
the function decompress_numpy_array. This structure does NOT have stable ABI, and
dedicated serialization format should be used. If the user wishes to decompress
a block that has not been a result of compress_numpy_array call (for example
read from a file) they can manually create a proper CompressedBlock instance.

data is the pointer to the raw compressed bytes
size is the size in bytes of the compressed data
num_entries is the number of entries that were compressed and should be decompressed.
    That is, the number of entries must be known both during compression and decompression.
'''
class CompressedBlock(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_uint8)),
        ('size', ctypes.c_size_t),
        ('num_entries', ctypes.c_size_t)
    ]

CompressedBlockPtr = ctypes.POINTER(CompressedBlock)

ac_destroy_compressed_block = dll.ac_destroy_compressed_block
ac_destroy_compressed_block.argtypes = [CompressedBlockPtr]

def make_ac_compress(entry_type, function_name):
    f = dll[function_name]
    f.restype = CompressedBlockPtr
    f.argtypes = [ctypes.POINTER(entry_type), ctypes.c_size_t]
    return f

def make_ac_decompress(entry_type, function_name):
    f = dll[function_name]
    f.restype = ctypes.POINTER(entry_type)
    f.argtypes = [CompressedBlockPtr]
    return f

def make_ac_destroy(entry_type, function_name):
    f = dll[function_name]
    f.argtypes = [ctypes.POINTER(entry_type)]
    return f

# COMPRESSION:

#     8 bit entries

ac_compress_u8_entry_u8_symbol = make_ac_compress(ctypes.c_uint8, 'ac_compress_u8_entry_u8_symbol')
ac_compress_i8_entry_u8_symbol = make_ac_compress(ctypes.c_int8, 'ac_compress_i8_entry_u8_symbol')

#     16 bit entries

ac_compress_u16_entry_u8_symbol = make_ac_compress(ctypes.c_uint16, 'ac_compress_u16_entry_u8_symbol')
ac_compress_u16_entry_u16_symbol = make_ac_compress(ctypes.c_uint16, 'ac_compress_u16_entry_u16_symbol')
ac_compress_i16_entry_u8_symbol = make_ac_compress(ctypes.c_int16, 'ac_compress_i16_entry_u8_symbol')
ac_compress_i16_entry_u16_symbol = make_ac_compress(ctypes.c_int16, 'ac_compress_i16_entry_u16_symbol')

#     32 bit entries

ac_compress_u32_entry_u8_symbol = make_ac_compress(ctypes.c_uint32, 'ac_compress_u32_entry_u8_symbol')
ac_compress_u32_entry_u16_symbol = make_ac_compress(ctypes.c_uint32, 'ac_compress_u32_entry_u16_symbol')
ac_compress_i32_entry_u8_symbol = make_ac_compress(ctypes.c_int32, 'ac_compress_i32_entry_u8_symbol')
ac_compress_i32_entry_u16_symbol = make_ac_compress(ctypes.c_int32, 'ac_compress_i32_entry_u16_symbol')

#     64 bit entries

ac_compress_u64_entry_u8_symbol = make_ac_compress(ctypes.c_uint64, 'ac_compress_u64_entry_u8_symbol')
ac_compress_u64_entry_u16_symbol = make_ac_compress(ctypes.c_uint64, 'ac_compress_u64_entry_u16_symbol')
ac_compress_i64_entry_u8_symbol = make_ac_compress(ctypes.c_int64, 'ac_compress_i64_entry_u8_symbol')
ac_compress_i64_entry_u16_symbol = make_ac_compress(ctypes.c_int64, 'ac_compress_i64_entry_u16_symbol')

# DECOMPRESSION:

#     8 bit entries

ac_decompress_u8_entry_u8_symbol = make_ac_decompress(ctypes.c_uint8, 'ac_decompress_u8_entry_u8_symbol')
ac_decompress_i8_entry_u8_symbol = make_ac_decompress(ctypes.c_int8, 'ac_decompress_i8_entry_u8_symbol')

#     16 bit entries

ac_decompress_u16_entry_u8_symbol = make_ac_decompress(ctypes.c_uint16, 'ac_decompress_u16_entry_u8_symbol')
ac_decompress_u16_entry_u16_symbol = make_ac_decompress(ctypes.c_uint16, 'ac_decompress_u16_entry_u16_symbol')
ac_decompress_i16_entry_u8_symbol = make_ac_decompress(ctypes.c_int16, 'ac_decompress_i16_entry_u8_symbol')
ac_decompress_i16_entry_u16_symbol = make_ac_decompress(ctypes.c_int16, 'ac_decompress_i16_entry_u16_symbol')

#     32 bit entries

ac_decompress_u32_entry_u8_symbol = make_ac_decompress(ctypes.c_uint32, 'ac_decompress_u32_entry_u8_symbol')
ac_decompress_u32_entry_u16_symbol = make_ac_decompress(ctypes.c_uint32, 'ac_decompress_u32_entry_u16_symbol')
ac_decompress_i32_entry_u8_symbol = make_ac_decompress(ctypes.c_int32, 'ac_decompress_i32_entry_u8_symbol')
ac_decompress_i32_entry_u16_symbol = make_ac_decompress(ctypes.c_int32, 'ac_decompress_i32_entry_u16_symbol')

#     64 bit entries

ac_decompress_u64_entry_u8_symbol = make_ac_decompress(ctypes.c_uint64, 'ac_decompress_u64_entry_u8_symbol')
ac_decompress_u64_entry_u16_symbol = make_ac_decompress(ctypes.c_uint64, 'ac_decompress_u64_entry_u16_symbol')
ac_decompress_i64_entry_u8_symbol = make_ac_decompress(ctypes.c_int64, 'ac_decompress_i64_entry_u8_symbol')
ac_decompress_i64_entry_u16_symbol = make_ac_decompress(ctypes.c_int64, 'ac_decompress_i64_entry_u16_symbol')

# DELETION:

ac_destroy_entries_u8 = make_ac_destroy(ctypes.c_uint8, 'ac_destroy_entries_u8')
ac_destroy_entries_u16 = make_ac_destroy(ctypes.c_uint16, 'ac_destroy_entries_u16')
ac_destroy_entries_u32 = make_ac_destroy(ctypes.c_uint32, 'ac_destroy_entries_u32')
ac_destroy_entries_u64 = make_ac_destroy(ctypes.c_uint64, 'ac_destroy_entries_u64')
ac_destroy_entries_i8 = make_ac_destroy(ctypes.c_int8, 'ac_destroy_entries_i8')
ac_destroy_entries_i16 = make_ac_destroy(ctypes.c_int16, 'ac_destroy_entries_i16')
ac_destroy_entries_i32 = make_ac_destroy(ctypes.c_int32, 'ac_destroy_entries_i32')
ac_destroy_entries_i64 = make_ac_destroy(ctypes.c_int64, 'ac_destroy_entries_i64')

def _compress_numpy_array_unchecked(array, symbol_type):
    if array.dtype == np.uint8:
        array_ctypes = array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if symbol_type == ctypes.c_uint8:
            return ac_compress_u8_entry_u8_symbol(array_ctypes, array.size)
    elif array.dtype == np.int8:
        array_ctypes = array.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        if symbol_type == ctypes.c_uint8:
            return ac_compress_i8_entry_u8_symbol(array_ctypes, array.size)
    elif array.dtype == np.uint16:
        array_ctypes = array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        if symbol_type == ctypes.c_uint8:
            return ac_compress_u16_entry_u8_symbol(array_ctypes, array.size)
        elif symbol_type == ctypes.c_uint16:
            return ac_compress_u16_entry_u16_symbol(array_ctypes, array.size)
    elif array.dtype == np.int16:
        array_ctypes = array.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        if symbol_type == ctypes.c_uint8:
            return ac_compress_i16_entry_u8_symbol(array_ctypes, array.size)
        elif symbol_type == ctypes.c_uint16:
            return ac_compress_i16_entry_u16_symbol(array_ctypes, array.size)
    elif array.dtype == np.uint32:
        array_ctypes = array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        if symbol_type == ctypes.c_uint8:
            return ac_compress_u32_entry_u8_symbol(array_ctypes, array.size)
        elif symbol_type == ctypes.c_uint16:
            return ac_compress_u32_entry_u16_symbol(array_ctypes, array.size)
    elif array.dtype == np.int32:
        array_ctypes = array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        if symbol_type == ctypes.c_uint8:
            return ac_compress_i32_entry_u8_symbol(array_ctypes, array.size)
        elif symbol_type == ctypes.c_uint16:
            return ac_compress_i32_entry_u16_symbol(array_ctypes, array.size)
    elif array.dtype == np.uint64:
        array_ctypes = array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        if symbol_type == ctypes.c_uint8:
            return ac_compress_u64_entry_u8_symbol(array_ctypes, array.size)
        elif symbol_type == ctypes.c_uint16:
            return ac_compress_u64_entry_u16_symbol(array_ctypes, array.size)
    elif array.dtype == np.int64:
        array_ctypes = array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        if symbol_type == ctypes.c_uint8:
            return ac_compress_i64_entry_u8_symbol(array_ctypes, array.size)
        elif symbol_type == ctypes.c_uint16:
            return ac_compress_i64_entry_u16_symbol(array_ctypes, array.size)

    raise Exception('Unsupported entry type {} or symbol type {}.'.format(array.dtype, symbol_type))

def _decompress_numpy_array_unchecked(compressed_block, entry_type, symbol_type):
    if entry_type == np.uint8:
        if symbol_type == ctypes.c_uint8:
            return ac_decompress_u8_entry_u8_symbol(compressed_block)
    elif entry_type == np.int8:
        if symbol_type == ctypes.c_uint8:
            return ac_decompress_i8_entry_u8_symbol(compressed_block)
    elif entry_type == np.uint16:
        if symbol_type == ctypes.c_uint8:
            return ac_decompress_u16_entry_u8_symbol(compressed_block)
        elif symbol_type == ctypes.c_uint16:
            return ac_decompress_u16_entry_u16_symbol(compressed_block)
    elif entry_type == np.int16:
        if symbol_type == ctypes.c_uint8:
            return ac_decompress_i16_entry_u8_symbol(compressed_block)
        elif symbol_type == ctypes.c_uint16:
            return ac_decompress_i16_entry_u16_symbol(compressed_block)
    elif entry_type == np.uint32:
        if symbol_type == ctypes.c_uint8:
            return ac_decompress_u32_entry_u8_symbol(compressed_block)
        elif symbol_type == ctypes.c_uint16:
            return ac_decompress_u32_entry_u16_symbol(compressed_block)
    elif entry_type == np.int32:
        if symbol_type == ctypes.c_uint8:
            return ac_decompress_i32_entry_u8_symbol(compressed_block)
        elif symbol_type == ctypes.c_uint16:
            return ac_decompress_i32_entry_u16_symbol(compressed_block)
    elif entry_type == np.uint64:
        if symbol_type == ctypes.c_uint8:
            return ac_decompress_u64_entry_u8_symbol(compressed_block)
        elif symbol_type == ctypes.c_uint16:
            return ac_decompress_u64_entry_u16_symbol(compressed_block)
    elif entry_type == np.int64:
        if symbol_type == ctypes.c_uint8:
            return ac_decompress_i64_entry_u8_symbol(compressed_block)
        elif symbol_type == ctypes.c_uint16:
            return ac_decompress_i64_entry_u16_symbol(compressed_block)

    raise Exception('Unsupported entry type {} or symbol type {}.'.format(entry_type, symbol_type))

'''
Destroy the array pointed to by entries ctypes object.
entries is expected to be a result of a decompress_numpy_array call.
'''
def destroy_entries(entries):
    if entries._type_ == ctypes.c_int8:
        ac_destroy_entries_i8(entries)
    elif entries._type_ == ctypes.c_uint8:
        ac_destroy_entries_u8(entries)
    elif entries._type_ == ctypes.c_int16:
        ac_destroy_entries_i16(entries)
    elif entries._type_ == ctypes.c_uint16:
        ac_destroy_entries_u16(entries)
    elif entries._type_ == ctypes.c_int32:
        ac_destroy_entries_i32(entries)
    elif entries._type_ == ctypes.c_uint32:
        ac_destroy_entries_u32(entries)
    elif entries._type_ == ctypes.c_int64:
        ac_destroy_entries_i64(entries)
    elif entries._type_ == ctypes.c_uint64:
        ac_destroy_entries_u64(entries)
    else:
        raise Exception('Unsupported entry type {}.'.format(entries._type_))

'''
Destroys the CompressedBlock pointed to by compressed_block ctypes object.
compressed_block is expected to be a result of a compress_numpy_array call
'''
def destroy_compressed_block(compressed_block):
    ac_destroy_compressed_block(compressed_block)

'''
If possible, compresses the numpy array. The numpy array might
contain entries of any integer type. The array must be c_contiguous.
symbol_type must be either ctypes.c_uint8 or ctypes.c_uint16.
The function throws Exception on failure, and may fail if the compressed
size exceeds the uncompressed size.
'''
def compress_numpy_array(array, symbol_type):
    if not array.data.c_contiguous:
        raise Exception('Numpy array not c_contiguous.')

    compressed = _compress_numpy_array_unchecked(array, symbol_type)
    if not compressed.contents.data or compressed.contents.size == 0:
        raise Exception('Compression failed.')

    return compressed

'''
Decompresses a CompressedBlock pointed by compressed_block ctypes object.
entry_type must be a numpy integer dtype.
symbol_type must be either ctypes.c_uint8 or ctypes.c_uint16.
On success it return a numpy array with elements of dtype=entry_type
and the number of elements equal to compressed_block.contents.num_entries.
Throws Exception on failure.
'''
def decompress_numpy_array(compressed_block, entry_type, symbol_type):
    decompressed = _decompress_numpy_array_unchecked(compressed_block, entry_type, symbol_type)

    if not decompressed:
        raise Exception('Decompression failed.')

    ret = np.ctypeslib.as_array(decompressed, shape=(compressed_block.contents.num_entries,)).copy()

    destroy_entries(decompressed)

    return ret

'''
Tests all compression and decompression overloads.
'''
def test():
    np.random.seed(123456)
    entry_types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64]
    ranges = [(0, 10), (-10, 10), (0, 100), (-10, 100), (0, 1000), (-10, 1000), (0, 10000), (-10, 10000)]
    symbol_types = [ctypes.c_uint8, ctypes.c_uint16]

    for entry_type, (low, high) in zip(entry_types, ranges):
        for symbol_type in symbol_types:
            if ctypes.sizeof(symbol_type) > np.dtype(entry_type).itemsize:
                continue

            print('Testing entry type {} and symbol type {}.'.format(entry_type, symbol_type))

            uncompressed = np.random.randint(low=low, high=high, size=(123, 456), dtype=entry_type)
            uncompressed = uncompressed * np.abs(uncompressed)
            uncompressed = uncompressed.flatten()

            print('Uncompressed size: {} {}'.format(uncompressed.size, uncompressed.size * uncompressed.itemsize))

            compressed = compress_numpy_array(uncompressed, symbol_type)

            print('Compressed size  : {}'.format(compressed.contents.size))

            decompressed = decompress_numpy_array(compressed, entry_type, symbol_type)

            print('Decompressed size: {} {}'.format(decompressed.size, uncompressed.size * decompressed.itemsize))

            for i in range(len(uncompressed)):
                if uncompressed[i] != decompressed[i]:
                    print('Value mismatch at {}. Uncompressed = {}, decompressed = {}.'.format(i, uncompressed[i], decompressed[i]))
                    raise Exception('')

            destroy_compressed_block(compressed)

            print('OK')

if __name__ == '__main__':
    test()
