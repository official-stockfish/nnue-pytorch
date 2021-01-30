import numpy as np
import enum
import struct
import io
import math

'''
General schema:

int64
    payload : byte[8]
        little endian

uint64
    payload : byte[8]
        little endian

int32
    payload : byte[4]
        little endian

uint32
    payload : byte[4]
        little endian

uint16
    payload : byte[2]
        little endian

int16
    payload : byte[2]
        little endian

uint8
    payload : byte

int8
    payload : byte

float16
    payload : byte[2]
        IEEE754

float32
    payload : byte[4]
        IEEE754

float64
    payload : byte[8]
        IEEE754

bool
    payload : byte
        0 - false
        1 - true
        any other value is not allowed

ShortString
    size : uint8
    payload : size bytes
        utf-8

LongString
    size : uint16
    payload : size bytes
        utf-8

DataType
    enum : uint8
        0 - uint8
        1 - int8
        2 - uint16
        3 - int16
        4 - uint32
        5 - int32
        6 - uint64
        7 - int64
        8 - float16 (IEEE754)
        9 - float32 (IEEE754)
        10 - float64 (IEEE754)
        ...
        values from 11 to 255 are reserved

FeatureTransformerPerspectiveTransform
    enum : uint8
        0 - none
        1 - flip
        2 - rotate
        ...
        values from 3 to 255 are reserved

Layer<TName>
    field_size : uint32
        the number of bytes taken by this field, including the concrete layer, including this property
    layer_type_name = TName : ShortString

InputSlice : Layer<"InputSlice">
    in_data_type : DataType
    out_data_type : DataType
    offset : uint32
    size : uint32

ClippedReLU : Layer<"ClippedReLU">
    in_data_type : DataType
    out_data_type : DataType
    min_activation : in_data_type
    max_activation : in_data_type

AffineTransform : Layer<"AffineTransform">
    weight_data_type : DataType
    bias_data_type : DataType
    if weight_data_type is integer then
        output_scale : uint32
    end
    num_inputs : uint32
    num_outputs : uint32
    biases : bias_data_type[num_outputs]
    weights : weight_data_type[num_outputs][num_inputs]

FeatureBlockFlags
    enum : uint8
        0 = plain
        1 = compressed
        ...
        values from 2 to 255 are reserved

CompressedFeatureBlockRow<weight_data_type, num_real_features, num_virtual_features>
    vlq_block_sizes : uint8[4]
        There may be more than 4 blocks encoding one number.
        Every block after the 4th one is assumed to have the same
        size as the 4th one (that is vlq_block_sizes[3]).
    compressed_payload_num_bits : uint32
    compressed_payload : byte[ceil(compressed_payload_num_bits / 8)]
            after decompression it is weight_data_type[num_real_features+num_virtual_features]

FeatureBlockTypeName : ShortString
    available feature blocks:
        "HalfKP"
            41024 real features, 1+64+641 virtual if compression used
        "HalfKA"
            641*11 real features, 1+64+641+64 virtual if compression used
        ...
        TODO: Maybe redefine the halfkp features such that there's 640 instead of 641?

FeatureBlock<weight_data_type, num_outputs>
    field_size : uint32
        the number of bytes this feature block occupies, including this property
    feature_block_type_name : FeatureBlockTypeName
    num_real_features : uint32
    flags : FeatureBlockFlags
    if flags.compressed then
        num_virtual_features : uint32
        compressed_rows :
            CompressedFeatureBlockRow<
                weight_data_type,
                num_real_features,
                num_virtual_features
            >[num_outputs]
    else
        weights : weight_data_type[num_outputs][num_real_features]
    end

FeatureTransformer
    weight_data_type : DataType
    bias_data_type : DataType
    split_perspectives : bool
    if split_perspectives then
        perspective_transform : FeatureTransformerPerspectiveTransform
    end
    num_outputs : uint32
    num_feature_blocks : uint8
    biases : bias_data_type[num_outputs]
    feature_blocks : FeatureBlock<weight_data_type, num_outputs>[num_feature_blocks]

LayerStack
    num_layers : uint8
    layers : Layer[num_layers]

Net
    feature_transformer : FeatureTransformer
    num_layer_stacks : uint8
    layer_stacks : LayerStack[num_layer_stacks]

File
    "NNUE" : byte[4]
    version : uint8
    *reserved* : uint8
        must be 0
    header_size : uint16
        in bytes, including the "NNUE" marker
    num_authors : uint8
    authors : ShortString[num_authors]
    description : LongString
    num_nets : byte
    nets : Net[num_nets]
'''

def product(iterable):
    p = 1
    for v in iterable:
        p *= v
    return p

def assert_isinstance(value, t):
    if value is None:
        raise RuntimeError('Expected value of type {}, got None'.format(str(value.__class__)))

    if not isinstance(value, t):
        raise RuntimeError('Expected value of type {}, got {}'.format(str(value.__class__), str(t)))

def assert_less_or_equal(val, max_val, text=''):
    if val > max_val:
        raise RuntimeError('{} {} exceeds maximum {}'.format(text, val, max_val))

def assert_greater_or_equal(val, min_val, text=''):
    if val < min_val:
        raise RuntimeError('{} {} is lower than the minimum {}'.format(text, val, min_val))

def write_uint8(stream, v):
    '''
    Writes a single value of type uint8 to a byte stream.
    '''
    if v < 0 or v > 255:
        raise Exception('Value {} is out of range of uint8'.format(v))
    stream.write(struct.pack('B', v))

def write_uint16(stream, v):
    '''
    Writes a single value of type uint16 to a byte stream.
    The byte ordering is little-endian, as for all types.
    '''
    if v < 0 or v > 2**16-1:
        raise Exception('Value {} is out of range of uint16'.format(v))
    stream.write(struct.pack('<H', v))

def write_uint32(stream, v):
    '''
    Writes a single value of type uint32 to a byte stream.
    The byte ordering is little-endian, as for all types.
    '''
    if v < 0 or v > 2**32-1:
        raise Exception('Value {} is out of range of uint32'.format(v))
    stream.write(struct.pack('<I', v))

def read_uint8(stream):
    '''
    Reads a single value of type uint8 from a byte stream.
    '''
    return struct.unpack('B', stream.read(1))[0]

def read_uint16(stream):
    '''
    Reads a single value of type uint16 from a byte stream.
    The byte ordering is little-endian, as for all types.
    '''
    return struct.unpack('<H', stream.read(2))[0]

def read_uint32(stream):
    '''
    Reads a single value of type uint32 from a byte stream.
    The byte ordering is little-endian, as for all types.
    '''
    return struct.unpack('<I', stream.read(4))[0]

def write_small_string(stream, string):
    '''
    Writes a small string to a byte stream. A small string is a regular
    python string (str) but is required to have length not larger than 255.
    A small string is encoded as:
        size : uint8
        payload : byte[size]
    The payload must be a valid utf-8 string.
    '''
    assert_isinstance(string, str)

    data = string.encode(encoding='utf-8', errors='strict')

    assert_less_or_equal(len(data), 255, 'Short string length')

    write_uint8(stream, len(data))
    stream.write(data)

def get_small_string_size(string):
    '''
    Returns the size of an encoded small string.
    Does not validate the correctness of the string.
    '''
    data = string.encode(encoding='utf-8', errors='strict')
    return 1 + len(data)

def write_long_string(stream, string):
    '''
    Writes a large string to a byte stream. A large string is a regular
    python string (str) but is required to have length not larger than 65535.
    A small string is encoded as:
        size : uint16
        payload : byte[size]
    The payload must be a valid utf-8 string.
    '''
    assert_isinstance(string, str)

    data = string.encode(encoding='utf-8', errors='strict')

    assert_less_or_equal(len(data), 2**16-1, 'Long string length')

    write_uint16(stream, len(data))
    stream.write(data)

def get_long_string_size(string):
    '''
    Returns the size of an encoded long string.
    Does not validate the correctness of the string.
    '''
    data = string.encode(encoding='utf-8', errors='strict')
    return 2 + len(data)

def write_numpy_ndarray(stream, v):
    '''
    Writes a numpy ndarray to a byte stream. The byte order is forced to little-endian.
    For multidimensional arrays the order is the same as in C.
    '''
    v = v.astype(v.dtype.newbyteorder('<'))
    stream.write(v.tobytes())

def read_numpy_ndarray(stream, dtype, shape):
    '''
    Reads a numpy ndarray from a byte stream. The byte order is forced to little-endian.
    For multidimensional arrays the order is the same as in C.
    '''
    size = product(shape)
    arr = np.frombuffer(stream.read(dtype.itemsize*size), dtype.newbyteorder('<'), size)
    arr = arr.reshape(shape).astype(dtype)
    return arr

def read_small_string(stream):
    '''
    Reads a small string from a byte stream.
    See `write_small_string` for the layout.
    '''
    size = read_uint8(stream)
    data = stream.read(size)
    if len(data) != size:
        raise Exception('Unexpected end of stream while reading small string')

    return data.decode(encoding='utf-8', errors='strict')

def read_long_string(stream):
    '''
    Reads a long string from a byte stream.
    See `write_long_string` for the layout.
    '''
    size = read_uint16(stream)
    data = stream.read(size)
    if len(data) != size:
        raise Exception('Unexpected end of stream while reading long string')

    return data.decode(encoding='utf-8', errors='strict')

def encode_signed_integer(v):
    '''
    Returns the signed integer encoded as an unsigned integer.
    This is useful because it avoids the two's complement encoding issues.
    The encoding is simple, it relies on putting the sign in the least significant bit
    and making the bits on the left encode the absolute value of the number.
    To use the whole space 1 is added to negative values before taking the absolute value.
    The procedure is best described by the code below.
    '''
    assert_isinstance(v, int)
    if v < 0:
        return ((-v - 1) << 1) | 1
    else:
        return v << 1

def decode_signed_integer(v):
    '''
    Returns the signed integer decoded from the passed unsigned integer.
    For the encoding scheme see `encode_signed_integer`
    '''
    assert_isinstance(v, int)
    if v < 0:
        raise Exception('Cannot decode a signed integer')

    if v & 1:
        return -((v >> 1) + 1)
    else:
        return v >> 1

class DataTypeItem:
    '''
    Represents a data type and its traits.
    '''
    def __init__(self, id, dtype, signed, group, sizeof):
        self.id = id
        self.dtype = dtype
        self.signed = signed
        self.group = group
        self.sizeof = sizeof

class DataType(enum.Enum):
    '''
    An enumeration of valid data types supported by the NNUE format.
    '''
    uint8 = DataTypeItem(id=0, dtype=np.dtype(np.uint8), signed=False, group='int', sizeof=1)
    int8 = DataTypeItem(id=1, dtype=np.dtype(np.int8), signed=True, group='int', sizeof=1)
    uint16 = DataTypeItem(id=2, dtype=np.dtype(np.uint16), signed=False, group='int', sizeof=2)
    int16 = DataTypeItem(id=3, dtype=np.dtype(np.int16), signed=True, group='int', sizeof=2)
    uint32 = DataTypeItem(id=4, dtype=np.dtype(np.uint32), signed=False, group='int', sizeof=4)
    int32 = DataTypeItem(id=5, dtype=np.dtype(np.int32), signed=True, group='int', sizeof=4)
    uint64 = DataTypeItem(id=6, dtype=np.dtype(np.uint64), signed=False, group='int', sizeof=8)
    int64 = DataTypeItem(id=7, dtype=np.dtype(np.int64), signed=True, group='int', sizeof=8)
    float16 = DataTypeItem(id=8, dtype=None, signed=True, group='float', sizeof=2)
    float32 = DataTypeItem(id=9, dtype=np.dtype(np.float32), signed=True, group='float', sizeof=4)
    float64 = DataTypeItem(id=10, dtype=np.dtype(np.float64), signed=True, group='float', sizeof=8)

    def from_numpy(dtype):
        for v in DataType:
            if v.value.dtype and v.value.dtype == dtype:
                return v
        raise RuntimeError('Unknown dtype {}'.format(dtype))

    def is_integer(self):
        return self.value.group == 'int'

    def to_numpy(self):
        return self.value.dtype

    def is_signed(self):
        return self.value.signed

    def sizeof(self):
        return self.value.sizeof

    def write(self, stream):
        write_uint8(stream, self.value.id)

    @staticmethod
    def read(stream):
        id = read_uint8(stream)
        for v in DataType:
            if v.value.id == id:
                return v
        raise Exception('Invalid DataType')

def write_typed_value(stream, value, data_type):
    '''
    Writes a single value of type `data_type` to the byte stream.
    '''
    assert_isinstance(data_type, DataType)
    write_numpy_ndarray(stream, np.array([value], dtype=data_type.to_numpy()))

def read_typed_value(stream, data_type):
    '''
    Reads a single value of type `data_type` from the byte stream.
    '''
    assert_isinstance(data_type, DataType)
    return read_numpy_ndarray(stream, data_type.to_numpy(), (1,))[0]

def get_histogram(data):
    '''
    Returns a dictionary with keys being unique values from the ndarray `data`
    and values being the number of occurences of the respective values.
    '''
    value, counts = np.unique(data.flatten(), return_counts=True)
    return {i : c for i, c in zip(value, counts)}

class BitStreamWriter:
    '''
    A stream that allows writing single bits.
    The order is from most significant bit to least significant bit.
    Multibit values can be serialized. Note that endianness doesn't matter.
    Every write is an unsigned integer and the number of bits to write.
    For example:
        Current state:
                                 cursor
                                 V
            10101010 01010101 111
            byte 0   byte 1   byte 3
        We want to append value 000011 00110011 (first 14 bits of the number 819)
        State after write:
                                 first new bit
                                 V
            10101010 01010101 11100001 10011001 1
            byte 0   byte 1   byte 2   byte 3   byte 4

    The lower unused bits of the last byte are always zero.
    '''
    def __init__(self):
        self.data = bytearray()
        self.unused_bits = 0
        self.num_bits = 0

    def write(self, v, num_bits):
        assert v >= 0
        self.num_bits += num_bits
        while num_bits > 0:
            v &= (1 << num_bits) - 1

            if self.unused_bits == 0:
                self.data.append(0)
                self.unused_bits = 8

            if self.unused_bits >= num_bits:
                shift = self.unused_bits - num_bits
                self.data[-1] |= v << shift
                self.unused_bits -= num_bits
                break
            else:
                shift = num_bits - self.unused_bits
                self.data[-1] |= v >> shift
                num_bits -= self.unused_bits
                self.unused_bits = 0

class BitStreamReader:
    '''
    A stream that allows reading single bits.
    Every read is parameterized by the number of bits to read.
    The result is an unsigned integer.
    For the schema see `BitStreamWriter`
    '''
    def __init__(self, data):
        self.data = data
        self.offset = 0
        self.unused_bits = 8

    def read(self, num_bits):
        v = 0
        while num_bits > 0:
            if self.unused_bits == 0:
                self.unused_bits = 8
                self.offset += 1

            if self.unused_bits >= num_bits:
                shift = self.unused_bits - num_bits
                part = ((self.data[self.offset] >> shift) & ((1 << num_bits) - 1))
                v = (v << num_bits) | part
                self.unused_bits -= num_bits
                break
            else:
                part = (self.data[self.offset] & ((1 << self.unused_bits) - 1))
                v = (v << self.unused_bits) | part
                num_bits -= self.unused_bits
                self.unused_bits = 0

        return v

class FeatureBlockFlags(enum.Flag):
    '''
    The flags for a feature block. Each flag occupies one bit.
    All bits that don't correspond to any flag must be 0.
    '''

    '''
    No flags
    '''
    plain = 0

    '''
    The feature block's weights are compressed. See `FeatureBlock` for more details.
    '''
    compressed = 1

    def write(self, stream):
        '''
        Writes the flags to the byte stream. The flags occupy one byte.
        '''
        write_uint8(stream, self.value)

    @staticmethod
    def read(stream):
        '''
        Reads the falgs from the byte stream. The flags occupy one byte.
        '''
        id = read_uint8(stream)
        for v in FeatureBlockFlags:
            if v.value == id:
                return v
        raise Exception('Invalid FeatureBlockFlags')

class FeatureTransformerPerspectiveTransform(enum.Enum):
    '''
    The enumeration of valid perspective transforms for the feature transformer.
    '''

    '''
    No transformation for the black perspective. That is, feature encoding is not
    dependent on the perspective.
    '''
    none = 0

    '''
    The board is flipped vertically when seen from black's perspective.
    '''
    flip = 1

    '''
    The board is rotated by 180 degrees when seen from black's perspective.
    '''
    rotate = 2

    def write(self, stream):
        '''
        Writes the enumeration to the byte stream. The enumeration occupy one byte.
        '''
        write_uint8(stream, self.value)

    @staticmethod
    def read(stream):
        '''
        Reads the enumeration from the byte stream. The enumeration occupy one byte.
        '''
        id = read_uint8(stream)
        for v in FeatureTransformerPerspectiveTransform:
            if v.value == id:
                return v
        raise Exception('Invalid FeatureTransformerPerspectiveTransform')

'''
The layer types recognized by this tool.
Internal implementation details.
'''
LAYER_TYPES = dict()

class UnknownLayer:
    '''
    A layer of type that is not recognized by this tool.
    Such layers can still be deserialized and serialized because
    the field size is encoded. This layer stores just the raw bytes
    from the stream.
    Additionally the actual name of the layer type is stored.
    '''
    def __init__(self, layer_type_name, data):
        self.layer_type_name = layer_type_name
        self.raw_data = raw_data

    def write(self, stream):
        field_size = len(self.raw_data) + get_small_string_size(self.layer_type_name)
        write_uint32(stream, field_size)
        write_small_string(stream, self.layer_type_name)
        stream.write(self.raw_data)

    @staticmethod
    def read(stream):
        layer = UnknownLayer()
        field_size = read_uint32(stream)
        layer.layer_type_name = read_small_string(stream)
        raw_data_size = field_size - DataType.uint32.sizeof() - get_small_string_size(layer.layer_type_name)
        layer.raw_data = stream.read(raw_data_size)
        return layer

class Layer:
    '''
    Base class for any serializable/deserializable layer.

    Schema:
        Layer<TName>
            field_size : uint32
                the number of bytes this field occupies (including the concrete layer data and this property)
            layer_type_name = TName : ShortString
    '''
    def __init__(self, layer_type_name):
        self.layer_type_name = layer_type_name

    def write(self, stream, child_stream):
        '''
        This is supposed to be called only from the derived class.
        The `child_stream` is assumed to contain the serialized concrete layer.
        The `stream` is the actual output byte stream
        '''
        child_stream.seek(0)
        data = child_stream.read()
        field_size = len(data) + DataType.uint32.sizeof() + get_small_string_size(self.layer_type_name)

        write_uint32(stream, field_size)
        write_small_string(stream, self.layer_type_name)
        stream.write(data)

    @staticmethod
    def read(stream):
        '''
        Reads a single layer from the input byte stream.
        If the layer type is not recognized it will be deserialized
        into an `UnknownLayer`.
        '''
        field_size = read_uint32(stream)
        print('Field size: {}'.format(field_size))
        layer_type_name = read_small_string(stream)

        if layer_type_name in LAYER_TYPES:
            layer_type = LAYER_TYPES[layer_type_name]
            return layer_type.read(stream)
        else:
            return UnknownLayer.read(stream)

class InputSlice(Layer):
    '''
    An input slice layer. This is usually the first layer after the feature transformer.
    It serves as an input to a layer stack, possible spanning outputs
    from both feature transformer perspectives.
    The input and output types might differ, in which case a conversion happens.
    int -> int
        overflow is undefined behaviour
    int -> float
        regular int to float conversion
    float -> int
        the value is rounded to the nearest integer
        if the input value is outside of the range the behaviour is undefined behaviour

    Schema:
        InputSlice : Layer<"InputSlice">
            in_data_type : DataType
            out_data_type : DataType
            offset : uint32
            size : uint32
    '''
    def __init__(self):
        super(InputSlice, self).__init__('InputSlice')

        self.in_data_type = None
        self.out_data_type = None
        self.offset = None
        self.size = None

    def write(self, stream):
        local_stream = io.BytesIO()

        self.in_data_type.write(local_stream)
        self.out_data_type.write(local_stream)
        write_uint32(local_stream, self.offset)
        write_uint32(local_stream, self.size)

        Layer.write(self, stream, local_stream)

    @staticmethod
    def read(stream):
        input_slice = InputSlice()

        input_slice.in_data_type = DataType.read(stream)
        input_slice.out_data_type = DataType.read(stream)
        input_slice.offset = read_uint32(stream)
        input_slice.size = read_uint32(stream)

        return input_slice

LAYER_TYPES['InputSlice'] = InputSlice

class ClippedReLU(Layer):
    '''
    A clipped ReLU layer. Like ReLU but also bounded from above.
    It effectively performs clamp(input, self.min_activation, self.max_activation) for each input
    The input and output types might differ, in which case a conversion happens.
    int -> int
        overflow is undefined behaviour
    int -> float
        regular int to float conversion
    float -> int
        the value is rounded to the nearest integer
        if the input value is outside of the range the behaviour is undefined behaviour
    NOTE: The conversion is performed after clipping.

    Schema:
        ClippedReLU : Layer<"ClippedReLU">
            in_data_type : DataType
            out_data_type : DataType
            min_activation : in_data_type
            max_activation : out_data_type
    '''
    def __init__(self):
        super(ClippedReLU, self).__init__('ClippedReLU')

        self.in_data_type = None
        self.out_data_type = None
        self.min_activation = None
        self.max_activation = None

    def write(self, stream):
        local_stream = io.BytesIO()

        self.in_data_type.write(local_stream)
        self.out_data_type.write(local_stream)
        write_typed_value(local_stream, self.min_activation, self.in_data_type)
        write_typed_value(local_stream, self.max_activation, self.in_data_type)

        Layer.write(self, stream, local_stream)

    @staticmethod
    def read(stream):
        crelu = ClippedReLU()

        crelu.in_data_type = DataType.read(stream)
        crelu.out_data_type = DataType.read(stream)
        crelu.min_activation = read_typed_value(stream, crelu.in_data_type)
        crelu.max_activation = read_typed_value(stream, crelu.in_data_type)

        return crelu

LAYER_TYPES['ClippedReLU'] = ClippedReLU

class AffineTransform(Layer):
    '''
    A fully connected layer.
    Effectively performs Y = Ax + b, where
        A is the weight matrix with shape [R, C]
        x is the input column vector of size C
        Y is the output column vector of size R
        b is the bias column vector of size R

    When performed in the integer domain and additional output_scale is specified
    by which all outputs are be divided.

    # TODO: Specify what happens when types differ

    Schema:
        AffineTransform : Layer<"AffineTransform">
            weight_data_type : DataType
            bias_data_type : DataType
            if weight_data_type is integer then
                output_scale : uint32
            end
            num_inputs : uint32
            num_outputs : uint32
            biases : bias_data_type[num_outputs]
            weights : weight_data_type[num_outputs][num_inputs]
    '''
    def __init__(self):
        super(AffineTransform, self).__init__('AffineTransform')

        self.output_scale = None
        self.biases = None
        self.weights = None

    def write(self, stream):
        local_stream = io.BytesIO()

        assert_isinstance(self.weights, np.ndarray)
        assert_isinstance(self.biases, np.ndarray)

        if len(self.weights.shape) != 2:
            raise RuntimeError('Affine transform weights must be 2-dimensional')

        if len(self.biases.shape) != 1:
            raise RuntimeError('Affine transform biases must be 1-dimensional')

        weight_data_type = DataType.from_numpy(self.weights.dtype)
        bias_data_type = DataType.from_numpy(self.biases.dtype)

        weight_data_type.write(local_stream)
        bias_data_type.write(local_stream)

        if weight_data_type.is_integer():
            assert_isinstance(self.output_scale, int)
            assert_greater_or_equal(self.output_scale, 1, 'Output scale')
            write_uint32(local_stream, self.output_scale)

        num_outputs = self.weights.shape[0]
        num_inputs = self.weights.shape[1]

        if self.biases.shape[0] != num_outputs:
            raise RuntimeError('Biases size does not match the weights size')

        write_uint32(local_stream, num_inputs)
        write_uint32(local_stream, num_outputs)

        write_numpy_ndarray(local_stream, self.biases)
        write_numpy_ndarray(local_stream, self.weights)

        Layer.write(self, stream, local_stream)

    @staticmethod
    def read(stream):
        layer = AffineTransform()
        weight_data_type = DataType.read(stream)
        bias_data_type = DataType.read(stream)

        if weight_data_type.is_integer():
            layer.output_scale = read_uint32(stream)

        num_inputs = read_uint32(stream)
        num_outputs = read_uint32(stream)

        layer.biases = read_numpy_ndarray(stream, bias_data_type.to_numpy(), (num_outputs,))
        layer.weights = read_numpy_ndarray(stream, weight_data_type.to_numpy(), (num_outputs, num_inputs))

        return layer

LAYER_TYPES['AffineTransform'] = AffineTransform

def get_vlq_encoded_size(histogram, block_sizes):
    '''
    Returns the number of bits requires to encode the values
    with the given histogram and the given block sizes.
    VLQ expands to "Variable Length Quantity", which is used for clarity.
    The encoding is actually a variable width encoding as described here
    https://en.wikipedia.org/wiki/Variable-width_encoding
    but with block sizes being of possibly different sizes.
    The block size is the number of coding bits within one block,
    excluding the bit required to designate end or continuation.
    '''
    def get_length(c):
        length = 0
        c = encode_signed_integer(int(c))
        bound = 0
        bits = 0
        block_id = 0
        while True:
            block_size = block_sizes[block_id]
            bits += block_size
            length += block_size + 1
            bound = 2**bits
            if c < bound:
                break
            if block_id + 1 < len(block_sizes):
                block_id += 1
        if c >= bound:
            raise Exception('Not enough space to encode value')
        return length

    return sum(get_length(c) * v for c, v in histogram.items())

def get_best_vlq_encoding_parameters(values):
    '''
    Returns 'the best' block sizes for encoding the given values.

    # TODO: Better and faster algorithm. Configurable number of blocks.
    '''
    if len(values.shape) != 1:
        raise Exception('Must be a 1d array for now')

    histogram = get_histogram(values)
    best = None
    for i in range(2, 8):
        for j in range(1, 4):
            for k in range(1, 4):
                for l in range(1, 4):
                    params = [i, j, k, l]
                    size = math.ceil(get_vlq_encoded_size(histogram, params))
                    if not best or size < best[0]:
                        best = (size, params)

    return best[1], best[0]

def vlq_encode_array(bit_stream, data, block_sizes):
    '''
    Encodes the data array using a variable width encoding with the
    given block sizes and writes it to the bit stream.
    Only integers can be encoded.
    '''
    nnue_dtype = DataType.from_numpy(data.dtype)
    if not nnue_dtype.is_integer():
        raise Exception('Cannot encode non-integer values')

    is_signed = nnue_dtype.is_signed()
    for v in data:
        if is_signed:
            v = encode_signed_integer(int(v))

        i = 0
        while True:
            block_size = block_sizes[i]
            bit_stream.write(v, block_size)
            v >>= block_size
            if i + 1 < len(block_sizes):
                i += 1

            if v == 0:
                bit_stream.write(0, 1)
                break
            else:
                bit_stream.write(1, 1)

def vlq_decode_array(bit_stream, dtype, block_sizes, size):
    '''
    Decodes the array of the given size from the bit stream
    using variable width encoding with given block sizes.
    The resulting array contains elements of type dtype.
    Only integers can be encoded/decoded.
    '''
    nnue_dtype = DataType.from_numpy(dtype)
    if not nnue_dtype.is_integer():
        raise Exception('Cannot decode non-integer values')

    is_signed = nnue_dtype.is_signed()
    data = []
    for i in range(size):
        b = 0
        off = 0
        v = 0
        while True:
            block_size = block_sizes[b]
            v |= bit_stream.read(block_size) << off

            e = bit_stream.read(1)
            if e == 0:
                break

            off += block_sizes[b]

            if b + 1 < len(block_sizes):
                b += 1

        if is_signed:
            v = decode_signed_integer(int(v))

        data.append(v)

    return np.array(data, dtype=dtype)

def compress_integer_array(data):
    '''
    Returns a bytes object with data encoded using variable width encoding.
    The blob contains information necessary to decode the data.
    Block sizes must not be smaller than 1 nor greater than 64.
    Only integers can be compressed/decompressed.

    Schema:
        CompressedIntegerArray:
            vlq_block_sizes : uint8[4]
                There may be more than 4 blocks encoding one number.
                Every block after the 4th one is assumed to have the
                same size as the 4th one (that is vlq_block_sizes[3]).
            compressed_payload_num_bits : uint32
            compressed_payload : byte[ceil(compressed_payload_num_bits / 8)]
                After decompression it is dtype[data.shape...]
    '''
    params, s = get_best_vlq_encoding_parameters(data)
    header = io.BytesIO()
    for i in range(4):
        block_size = params[i]
        assert_greater_or_equal(block_size, 1, 'Block size')
        assert_less_or_equal(block_size, 64, 'Block size')
        write_uint8(header, block_size)

    bit_stream = BitStreamWriter()
    vlq_encode_array(bit_stream, data, params)

    write_uint32(header, bit_stream.num_bits)

    header.seek(0)
    return header.read() + bit_stream.data

def decompress_integer_array(stream, dtype, size):
    '''
    Returns a numpy ndarray the specified number of decoded values
    from the given byte stream. The returned array is one
    dimensional and has elements of type dtype.
    Only integers can be compressed/decompressed.
    '''
    block_sizes = []
    for i in range(4):
        block_size = read_uint8(stream)
        assert_greater_or_equal(block_size, 1, 'Block size')
        assert_less_or_equal(block_size, 64, 'Block size')
        block_sizes.append(block_size)

    num_bits = read_uint32(stream)
    num_bytes = (num_bits + 7) // 8

    bit_stream = BitStreamReader(stream.read(num_bytes))
    return vlq_decode_array(bit_stream, dtype, block_sizes, size)

def compress_factorized_weights(factorized_weights):
    '''
    Returns a bytes object containing the compressed input array.
    The input is assumed to be a valid
    factorized feature transformer weights matrix.
    '''
    if len(factorized_weights.shape) != 2:
        raise Exception('Weights for compression must be 2d')

    factorized_weights = factorized_weights.transpose()
    # Now it's of shape [num_outputs][num_total_features]

    encoded = []
    for i in range(len(factorized_weights)):
        print('Row {}...'.format(i))
        row = factorized_weights[i]
        encoded.append(compress_integer_array(row))

    return b''.join(encoded)

def decompress_factorized_weights(stream, dtype, num_total_features, num_outputs):
    '''
    Decompresses a 2d integer ndarray from the stream.
    The resulting element type is `dtype`
    The resulting shape is [num_total_features][num_outputs]
    '''
    rows = []
    for i in range(num_outputs):
        print('Row {}...'.format(i))
        row = decompress_integer_array(stream, dtype, num_total_features)
        rows.append(row)

    return np.stack(rows).transpose()

class FeatureBlockDefinition:
    '''
    Used to store some basic traits of a feature block.
    '''
    def __init__(self, name, num_real_features, num_virtual_features):
        self.name = name
        self.num_real_features = num_real_features
        self.num_virtual_features = num_virtual_features

class HalfKP(FeatureBlockDefinition):
    '''
    Represents a HalfKP feature block.
    The virtual features defined for the purpose of compression are:
        1 connecting to all real features
        641 for 1 + piece_type * piece_position
        64 for king positions

    TODO: Revise and decide whether to use 641 or 640. One is unused.
    '''
    NUM_SQUARES = 64
    NUM_K_FEATURES = NUM_SQUARES
    NUM_PIECES = 10
    NUM_P_FEATURES = NUM_PIECES * NUM_SQUARES + 1
    NUM_REAL_FEATURES = NUM_K_FEATURES * NUM_P_FEATURES
    NUM_VIRTUAL_FEATURES = 1 + NUM_K_FEATURES + NUM_P_FEATURES

    def __init__(self):
        super(HalfKP, self).__init__('HalfKP', HalfKP.NUM_REAL_FEATURES, HalfKP.NUM_VIRTUAL_FEATURES)

    def factorize(self, i):
        return (\
            HalfKP.NUM_REAL_FEATURES, \
            HalfKP.NUM_REAL_FEATURES + 1 + i % HalfKP.NUM_P_FEATURES, \
            HalfKP.NUM_REAL_FEATURES + 1 + HalfKP.NUM_P_FEATURES + i // HalfKP.NUM_P_FEATURES \
        )

'''
The feature block types recognized by this tool.
Internal implementation details.
'''
FEATURE_BLOCK_DEFINITIONS = dict()

def add_feature_block_definition(cls):
    instance = cls()
    name = instance.name
    FEATURE_BLOCK_DEFINITIONS[name] = instance

add_feature_block_definition(HalfKP)

def factorize_weights(features_name, weights):
    '''
    Factorizes the given weights according to the factorizer for the
    given feature block type.
    The first dimension is the feature dimension.
    The factorization is performed in an attempt to minimize entropy.
    There's many possible factorizations and better methods
    might be found in the future.
    The factorization method is an implementation detail of the compressor.
    '''
    fb_def = FEATURE_BLOCK_DEFINITIONS[features_name]
    num_total_features = fb_def.num_real_features + fb_def.num_virtual_features

    if weights.shape[0] != fb_def.num_real_features:
        raise Exception('Expected {} features, got {}'.format(fb_def.num_real_features, weights.shape[0]))

    factorized_weights = np.zeros((num_total_features, weights.shape[1]), dtype=weights.dtype)

    part_of = [[] for i in range(num_total_features)]
    for i in range(fb_def.num_real_features):
        factorized_weights[i] = weights[i]

        for j in fb_def.factorize(i):
            part_of[j].append(i)

    for i, p in enumerate(part_of):
        if not p:
            continue

        assert i not in p
        assert p
        factorized_weights[i] = np.median(factorized_weights[p], axis=0)
        factorized_weights[p] -= factorized_weights[i]

    return factorized_weights

def coalesce_weights(features_name, factorized_weights):
    '''
    Performs an inverse of `factorize_weights`
    '''
    fb_def = FEATURE_BLOCK_DEFINITIONS[features_name]
    num_total_features = fb_def.num_real_features + fb_def.num_virtual_features

    if factorized_weights.shape[0] != num_total_features:
        raise Exception('Expected {} features, got {}'.format(num_total_features, factorized_weights.shape[0]))

    coalesced = np.zeros((fb_def.num_real_features, factorized_weights.shape[1]), dtype=factorized_weights.dtype)

    for i in range(fb_def.num_real_features):
        coalesced[i] = factorized_weights[i]
        for j in fb_def.factorize(i):
            coalesced[i] += factorized_weights[j]

    return coalesced

class FeatureBlock:
    '''
    Represents weights for a single feature block within the feature transformer.
    The weights can be stored either compressed or not (but in memory always uncompressed)
    The compression is only available for integer weights.
    If the feature block type is not recognized by the tool they are stored
    factorized in self.factorized_weights. This does not prevent editing the weights, but
    the user must understand how they are represented.

    Schema:
        FeatureBlock<weight_data_type, num_outputs>
            field_size : uint32
                the number of bytes used by this feature block, including this property
            feature_block_type_name : FeatureBlockTypeName
            num_real_features : uint32
            flags : FeatureBlockFlags
                see `FeatureBlockFlags`
            if flags.compressed then
                    num_virtual_features : uint32
                    compressed_rows :
                        CompressedFeatureBlockRow<
                            weight_data_type,
                            num_real_features,
                            num_virtual_features
                        >[num_outputs]
                        see `compress_factorized_weights`
            else
                    weights : weight_data_type[num_outputs][num_real_features]
            end
    '''
    def __init__(self):
        self.feature_block_type_name = None
        self.flags = None
        self.weights = None
        self.factorized_weights = None

    def write(self, stream):
        if len(self.weights.shape) != 2:
            raise RuntimeError('Feature transformer weights must be 2-dimensional')

        if self.feature_block_type_name in FEATURE_BLOCK_DEFINITIONS:
            fb_def = FEATURE_BLOCK_DEFINITIONS[self.feature_block_type_name]
            if self.weights.shape[0] != fb_def.num_real_features:
                raise Exception('Expected {} features, got {}'.format(fb_def.num_real_features, self.weights.shape[0]))

        # We need to write the field size first. That's why we use a local, temporary stream.
        local_stream = io.BytesIO()

        write_small_string(local_stream, self.feature_block_type_name)

        write_uint32(local_stream, self.weights.shape[0])

        assert_isinstance(self.flags, FeatureBlockFlags)
        self.flags.write(local_stream)

        if bool(self.flags & FeatureBlockFlags.compressed):
            self._write_compressed_weights(local_stream)
        else:
            self._write_uncompressed_weights(local_stream)

        local_stream.seek(0)
        data = local_stream.read()
        field_size = len(data) + DataType.uint32.sizeof()
        write_uint32(stream, field_size)
        stream.write(data)

    def _write_uncompressed_weights(self, stream):
        if self.weights is None:
            raise Exception('Unknown layer and weights are not coalesced')
        write_numpy_ndarray(stream, self.weights.transpose())

    def _write_compressed_weights(self, stream):
        if not DataType.from_numpy(self.weights.dtype).is_integer():
            raise Expected('Cannot compress non-integer weights')

        if self.weights is not None:
            factorized_weights = factorize_weights(self.feature_block_type_name, self.weights)
            write_uint32(stream, factorized_weights.shape[0] - self.weights.shape[0])
            stream.write(compress_factorized_weights(factorized_weights))
        else:
            write_uint32(stream, self.factorized_weights.shape[0])
            write_numpy_ndarray(self.factorized_weights.transpose())

    @staticmethod
    def read(stream, dtype, num_outputs):
        fb = FeatureBlock()

        field_size = read_uint32(stream)
        print('Field size: {}'.format(field_size))

        fb.feature_block_type_name = read_small_string(stream)
        print('Feature block type name: {}'.format(fb.feature_block_type_name))

        num_real_features = read_uint32(stream)
        if fb.feature_block_type_name in FEATURE_BLOCK_DEFINITIONS:
            fb_def = FEATURE_BLOCK_DEFINITIONS[fb.feature_block_type_name]
            if num_real_features != fb_def.num_real_features:
                raise Exception('Mismatching num_real_features')

        print('Number of features: {}'.format(num_real_features))

        fb.flags = FeatureBlockFlags.read(stream)
        if bool(fb.flags & FeatureBlockFlags.compressed):
            fb._read_compressed_weights(stream, dtype, num_real_features, num_outputs)
        else:
            fb._read_uncompressed_weights(stream, dtype, num_real_features, num_outputs)

        return fb

    def _read_compressed_weights(self, stream, dtype, num_real_features, num_outputs):
        if not DataType.from_numpy(dtype).is_integer():
            raise Expected('Cannot compress non-integer weights')

        num_virtual_features = read_uint32(stream)
        num_total_features = num_real_features + num_virtual_features

        if self.feature_block_type_name in FEATURE_BLOCK_DEFINITIONS:
            fb_def = FEATURE_BLOCK_DEFINITIONS[self.feature_block_type_name]
            if num_real_features != fb_def.num_real_features:
                raise Exception('Expected {} real features but got {}'.format(fb_def.num_real_features, num_real_features))
            if num_virtual_features != fb_def.num_virtual_features:
                raise Exception('Expected {} virtual features but got {}'.format(fb_def.num_virtual_features, num_virtual_features))

            factorized_weights = decompress_factorized_weights(stream, dtype, num_total_features, num_outputs)
            self.weights = coalesce_weights(self.feature_block_type_name, factorized_weights)
        else:
            self.factorized_weights = read_numpy_ndarray(stream, dtype, (num_outputs, num_total_features)).transpose()

    def _read_uncompressed_weights(self, stream, dtype, num_real_features, num_outputs):
        self.weights = read_numpy_ndarray(stream, dtype, (num_outputs, num_real_features)).transpose()

class FeatureTransformer:
    '''
    The feature transformer. It's a (sparse) fully connected layer.
    It consists of individual feature blocks that are layed out sequentially.
    It has an optional perspective transform.
    If perspective transform is None then it means the feature transformer has only one perspective.
    Otherwise it has two perspectives: one for white, one for black.

    Schema:
        FeatureTransformer
            weight_data_type : DataType
            bias_data_type : DataType
            split_perspectives : bool
            if split_perspectives then
                perspective_transform : FeatureTransformerPerspectiveTransform
            end
            num_outputs : uint32
            num_feature_blocks : uint8
            biases : bias_data_type[num_outputs]
            feature_blocks : FeatureBlock<weight_data_type, num_outputs>[num_feature_blocks]
    '''
    def __init__(self):
        self.perspective_transform = None
        self.feature_blocks = []
        self.biases = None

    def write(self, stream):
        if len(self.feature_blocks) == 0:
            raise RuntimeError('No feature blocks')

        assert_isinstance(self.biases, np.ndarray)

        if len(self.biases.shape) != 1:
            raise RuntimeError('Feature transformer biases must be 1-dimensional')

        num_outputs = self.biases.shape[0]

        weight_data_type = DataType.from_numpy(self.feature_blocks[0].weights.dtype)
        bias_data_type = DataType.from_numpy(self.biases.dtype)

        weight_data_type.write(stream)
        bias_data_type.write(stream)

        if self.perspective_transform is None:
            write_uint8(stream, 0)
        else:
            assert_isinstance(self.perspective_transform, FeatureTransformerPerspectiveTransform)
            write_uint8(stream, 1)
            self.perspective_transform.write(stream)

        write_uint32(stream, num_outputs)
        write_uint8(stream, len(self.feature_blocks))

        write_numpy_ndarray(stream, self.biases)

        for feature_block in self.feature_blocks:
            fb_weights = feature_block.weights if feature_block.weights is not None else feature_block.factorized_weights

            assert_isinstance(fb_weights, np.ndarray)

            if len(fb_weights.shape) != 2:
                raise RuntimeError('Feature transformer weights must be 2-dimensional')

            if fb_weights.shape[1] != num_outputs:
                raise RuntimeError('Feature transformer weights size mismatch. Expected {}, got {}'.format(num_outputs, fb_weights.shape[1]))

            if DataType.from_numpy(fb_weights.dtype) != weight_data_type:
                raise RuntimeError('Feature transformer weights type mismatch. Expected {}, got {}'.format(weight_data_type, fb_weights.dtype))

            feature_block.write(stream)

    @staticmethod
    def read(stream):
        ft = FeatureTransformer()

        weight_data_type = DataType.read(stream)
        bias_data_type = DataType.read(stream)

        has_perspective_transform = read_uint8(stream) == 1
        print('Has perspective transform: {}'.format(has_perspective_transform))
        if has_perspective_transform:
            ft.perspective_transform = FeatureTransformerPerspectiveTransform.read(stream)

        num_outputs = read_uint32(stream)
        print('Num outputs: {}'.format(num_outputs))
        num_feature_blocks = read_uint8(stream)
        print('Num features blocks: {}'.format(num_feature_blocks))

        ft.biases = read_numpy_ndarray(stream, bias_data_type.to_numpy(), (num_outputs,))
        for i in range(num_feature_blocks):
            ft.feature_blocks.append(FeatureBlock.read(stream, weight_data_type.to_numpy(), num_outputs))

        return ft

class LayerStack:
    '''
    A layer stack is a stack of layers after the feature transformer.
    It usually starts with an input slice of some sort.
    The maximum number of layers is 255.

    Schema:
        LayerStack
            num_layers : uint8
            layers : Layer[num_layers]
    '''
    def __init__(self):
        self.layers = []

    def write(self, stream):
        write_uint8(stream, len(self.layers))

        for layer in self.layers:
            assert_isinstance(layer, Layer)

            layer.write(stream)

    @staticmethod
    def read(stream):
        layer_stack = LayerStack()
        num_layers = read_uint8(stream)
        print('Number of layers: {}'.format(num_layers))
        for i in range(num_layers):
            layer_stack.layers.append(Layer.read(stream))
        return layer_stack


class Net:
    '''
    A single network. One network consists of a feature transformer and
    any amount of layer stacks (might even be 0).
    The maximum number of layer stacks is 255.

    Schema:
        Net
            feature_transformer : FeatureTransformer
            num_layer_stacks : uint8
            layer_stacks : LayerStack[num_layer_stacks]
    '''
    def __init__(self):
        self.feature_transformer = None
        self.layer_stacks = []

    def write(self, stream):
        if not self.feature_transformer:
            raise RuntimeError('No feature transformer')

        assert_isinstance(self.feature_transformer, FeatureTransformer)

        self.feature_transformer.write(stream)

        write_uint8(stream, len(self.layer_stacks))

        for layer_stack in self.layer_stacks:
            assert_isinstance(layer_stack, LayerStack)

            layer_stack.write(stream)

    @staticmethod
    def read(stream):
        net = Net()
        net.feature_transformer = FeatureTransformer.read(stream)

        num_layer_stacks = read_uint8(stream)
        print('Number of layer stacks: {}'.format(num_layer_stacks))
        for i in range(num_layer_stacks):
            net.layer_stacks.append(LayerStack.read(stream))

        return net

def write(stream, nets, authors=[], description=''):
    '''
    Writes the given networks along with other metadata into a single byte stream.

    Schema:
        "NNUE" : byte[4]
        version : uint8
        *reserved* : uint8
            must be 0
        header_size : uint16
             in bytes, including the "NNUE" marker
        num_authors : uint8
        authors : ShortString[num_authors]
        description : LongString
        num_nets : byte
        nets : Net[num_nets]
    '''
    version = 1

    header_size = \
        4 + \
        DataType.uint8.sizeof() + \
        DataType.uint8.sizeof() + \
        DataType.uint16.sizeof() + \
        DataType.uint8.sizeof() + \
        sum(get_small_string_size(author) for author in authors) + \
        get_long_string_size(description) + \
        DataType.uint8.sizeof()

    stream.write(b'NNUE')
    write_uint8(stream, version)
    stream.write(b'\0')
    write_uint16(stream, header_size)
    write_uint8(stream, len(authors))
    for author in authors:
        write_small_string(stream, author)

    write_long_string(stream, description)

    write_uint8(stream, len(nets))
    for net in nets:
        assert_isinstance(net, Net)
        net.write(stream)

def read(stream):
    '''
    Reads networks and metadata from the byte stream.
    '''
    preamble = stream.read(4)
    if preamble != b'NNUE':
        raise Exception('Invalid preamble. Expected b\'NNUE\'')

    version = read_uint8(stream)
    if version != 1:
        raise Exception('Invalid version: {}'.format(version))

    reserved_0 = read_uint8(stream)
    if reserved_0 != 0:
        raise Exception('Reserved field not zero')

    header_size = read_uint16(stream)

    num_authors = read_uint8(stream)
    authors = []
    for i in range(num_authors):
        authors.append(read_small_string(stream))

    description = read_long_string(stream)

    num_nets = read_uint8(stream)
    nets = []
    for i in range(num_nets):
        nets.append(Net.read(stream))

    return nets, authors, description

if __name__ == '__main__':
    import sys

    class NNUEReader():
      def __init__(self, f):
        self.f = f

        self.read_header()
        self.read_int32() # Feature transformer hash
        self.feature_transformer = self.read_feature_transformer()
        self.read_int32() # FC layers hash
        self.l0 = self.read_fc_layer((32, 512))
        self.l1 = self.read_fc_layer((32, 32))
        self.l2 = self.read_fc_layer((1, 32), is_output=True)

      def read_header(self):
        self.read_int32() # version
        self.read_int32() # halfkp network hash
        desc_len = self.read_int32() # Network definition
        description = self.f.read(desc_len)

      def tensor(self, dtype, shape):
        i = 1
        for s in shape:
          i *= s
        d = np.fromfile(self.f, dtype, i)
        d = d.reshape(shape)
        return d

      def read_feature_transformer(self):
        biases = self.tensor(np.int16, (256,))
        weights = self.tensor(np.int16, (41024, 256))
        return biases, weights

      def read_fc_layer(self, shape, is_output=False):
        biases = self.tensor(np.int32, (shape[0],))
        weights = self.tensor(np.int8, shape)
        return biases, weights

      def read_int32(self, expected=None):
        v = struct.unpack("<i", self.f.read(4))[0]
        if expected is not None and v != expected:
          raise Exception("Expected: %x, got %x" % (expected, v))
        return v

      def __str__(self):
        return 'FT: {}\nL0: {}\nL1: {}\nL2: {}\n'.format(self.feature_transformer[1].shape, self.l0[1].shape, self.l1[1].shape, self.l2[1].shape)

    with open(sys.argv[1], 'rb') as f:
        with open(sys.argv[2], 'wb') as fout:
            net = NNUEReader(f)
            print(str(net))

            net2 = Net()
            net2.feature_transformer = FeatureTransformer()
            net2.feature_transformer.perspective_transform = FeatureTransformerPerspectiveTransform.rotate
            net2.feature_transformer.biases = net.feature_transformer[0]
            #net2.feature_transformer.biases = net.feature_transformer[0][0:1]
            fb = FeatureBlock()
            fb.feature_block_type_name = 'HalfKP'
            fb.flags = FeatureBlockFlags.compressed if '-c' in sys.argv else FeatureBlockFlags.plain
            fb.weights = net.feature_transformer[1]
            #fb.weights = net.feature_transformer[1][:,0:1]
            net2.feature_transformer.feature_blocks = [fb]
            ls = LayerStack()
            ls.layers = [InputSlice(), ClippedReLU(), AffineTransform(), ClippedReLU(), AffineTransform(), ClippedReLU(), AffineTransform()]

            ls.layers[0].in_data_type = DataType.int16
            ls.layers[0].out_data_type = DataType.int16
            ls.layers[0].offset = 0
            ls.layers[0].size = 512

            ls.layers[1].in_data_type = DataType.int16
            ls.layers[1].out_data_type = DataType.int8
            ls.layers[1].min_activation = 0
            ls.layers[1].max_activation = 127

            ls.layers[2].output_scale = 64
            ls.layers[2].biases = net.l0[0]
            ls.layers[2].weights = net.l0[1]

            ls.layers[3].in_data_type = DataType.int32
            ls.layers[3].out_data_type = DataType.int8
            ls.layers[3].min_activation = 0
            ls.layers[3].max_activation = 127

            ls.layers[4].output_scale = 64
            ls.layers[4].biases = net.l1[0]
            ls.layers[4].weights = net.l1[1]

            ls.layers[5].in_data_type = DataType.int32
            ls.layers[5].out_data_type = DataType.int8
            ls.layers[5].min_activation = 0
            ls.layers[5].max_activation = 127

            ls.layers[6].output_scale = 16
            ls.layers[6].biases = net.l2[0]
            ls.layers[6].weights = net.l2[1]

            net2.layer_stacks = [ls]

            with io.BytesIO() as out:
                write(out, [net2], ['author1', 'author2'], 'description')
                out.seek(0)
                serialized = out.read()
                fout.write(serialized)

                if '--test' in sys.argv:
                    print('Serialized size: {}'.format(len(serialized)))
                    nets, authors, description = read(io.BytesIO(serialized))
                    with io.BytesIO() as out2:
                        write(out2, nets, authors, description)
                        out2.seek(0)
                        serialized2 = out2.read()
                        print('Reserialized size: {}'.format(len(serialized)))
                        assert serialized == serialized2
