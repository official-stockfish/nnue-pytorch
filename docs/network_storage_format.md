# Network Storage Format

## Expected use-case and design goals

The expected use case for this network storage format is to provide a simple and generic way of storing NNUE networks for Stockfish and other chess engines. The design goals are the following:

- minimal metadata
- support for (optional) compression
- single-pass parsable
- parsable with missing knowledge
- generic and easily extensible
- focus on the network's data instead of the network's semantics
- but still make semantics identifiable
- recoverable from partial corruption

## Format specification

All integers are stored in little-endian byte order. Signed integers are stored in two's complement.

### ShortString

- length : uint8_t
- payload : uint8_t[length]
    the payload is encoded with utf-8

### LongString

- length : uint16_t
- payload : uint8_t[length]
    the payload is encoded with utf-8

### DataType

- type : uint8_t
    - int8_t = 0
    - uint8_t = 1
    - int16_t = 2
    - uint16_t = 3
    - int32_t = 4
    - uint32_t = 5
    - int64_t = 6
    - uint64_t = 7
    - float16_ieee754 = 8
    - float32_ieee754 = 9
    - float64_ieee754 = 10
    - ShortString = 11

### Compression

- type : uint8_t
    - none = 0
    - arithmetic_coding = 1

### RawData<data_type, dimensions...\>

- data : data_type[dimensions...]
    - or a single scalar of data_type if rank == 0
    - stored in C order
    - no padding

### ArithmeticCodingCompressedData<data_type, dimensions...\>

- size : uint64_t
- data : uint8_t[size]
    - // TODO: Exact specification of the encoding and decoding.
    - supports any integer type
    - either 8 bit or 16 bit codes

### DataBlock

- "SFNN_BLOCK" : uint8_t[10]
    - magic used for idendification of a block in case of data corruption of a limited parser
- name : ShortString
    - "weight", "bias", "psqt"
- data_type : DataType
- compression : Compression
- rank : uint8_t
    - scalar has rank 0
    - vector has rank 1
    - matrix has rank 2
    - ..., any value that fits in uint8_t is allowed
- dimensions : uint32_t[rank]
    - only if rank > 0
- data : (if compression == none then RawData<data_type, dimensions...\>; elif compression == arithmetic_coding then ArithmeticCodingCompressedData<data_type, dimensions...\>)

### Header

- "SFNN_HEADER" : uint8_t[11]
    - magic used for idendification of a block in case of data corruption of a limited parser
- version : uint8_t
    - must be 0
- spdx_license : ShortString
    - see https://spdx.org/licenses/
- textual_network_semantics_identifier : LongString
    - it's infeasible to have a rigorous description of semantics
    - should contain a description of what this network should do, but doesn't need to be too detailed, just inter-layer semantics
    - for example "Integrated feature transformer material values, 8 buckets, chosen by ((piece_count - 1) / 4). 8 layer stacks, chosen by ((piece_count - 1) / 4). ClippedReLU before fully connected layers. WeightScale = 64. OutputScale = 16. Activation = 127." would describe the arch from [e8d64af1230fdac65bb0da246df3e7abe82e0838](https://github.com/official-stockfish/Stockfish/tree/e8d64af1230fdac65bb0da246df3e7abe82e0838)
- description : LongString
- num_authors : uint8_t
- authors : ShortString[num_authors]

### Layer

- "SFNN_LAYER" : uint8_t[11]
    - magic used for idendification of a block in case of data corruption of a limited parser
- name
    - for example "Stockfish::FullyConnected"
    - should be fairly precise, but should NOT contain the type of the held data
- num_data_blocks : uint8_t
    - degenerate layers are not allowed, i.e. must have at least one data block
    - semantics not encoded, so no representation for the InputSlice nor ClippedReLU
- data_blocks : DataBlock[num_data_blocks]

### Network

- header : Header
- layers : Layer[]
    - no a-priori layer count, because it would be kinda useless and might not be easy to know

## Name standardization

### Header.textual_network_semantics_identifier

- ["Integrated feature transformer material values, 8 buckets, chosen by ((piece_count - 1) / 4). 8 layer stacks, chosen by ((piece_count - 1) / 4). ClippedReLU before fully connected layers. WeightScale = 64. OutputScale = 16. Activation = 127."](https://github.com/official-stockfish/Stockfish/tree/e8d64af1230fdac65bb0da246df3e7abe82e0838)

### Layer.name

- ["Stockfish::FeatureTransformer"](https://github.com/official-stockfish/Stockfish/blob/3597f1942ec6f2cfbd50b905683739b0900ff5dd/src/nnue/nnue_feature_transformer.h#L83)
    - DataBlock.name: "feature_set", "weight", "bias"
        - feature_set: "HalfKP"
- ["Stockfish::FeatureTransformerWithPsqt"](https://github.com/official-stockfish/Stockfish/blob/773dff020968f7a6f590cfd53e8fd89f12e15e36/src/nnue/nnue_feature_transformer.h#L167)
    - DataBlock.name: "feature_set", "weight", "bias", "psqt"
        - feature_set: "HalfKAv2"
- ["Stockfish::FullyConnected"](https://github.com/official-stockfish/Stockfish/blob/773dff020968f7a6f590cfd53e8fd89f12e15e36/src/nnue/layers/affine_transform.h#L31)
    - DataBlock.name: "weight", "bias"
