#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <cstring>
#include <cstdint>
#include <utility>

#include "lib/export.h"

// Big endian is not supported by the trainer
const bool IsLittleEndian = true;

// This class represents an array with constant time known at runtime.
// The values stored must be of a trivial type. The whole array is
// zero-initialized on creation.
template <typename T>
struct DynArray
{
    static_assert(std::is_trivial_v<T>);

    // Create an array in empty state.
    DynArray() :
        data_(nullptr),
        size_(0)
    {
    }

    // Create a zero-initialized array with `size` elements.
    DynArray(std::size_t size) :
        data_(new T[size]),
        size_(size)
    {
        std::memset(data_, 0, size_ * sizeof(T));
    }

    DynArray(const DynArray&) = delete;
    DynArray(DynArray&& other) :
        data_(other.data_),
        size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DynArray& operator=(const DynArray&) = delete;
    DynArray& operator=(DynArray&& other)
    {
        if (data_ != nullptr)
            delete[] data_;

        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;

        return *this;
    }

          T* data()       { return data_; }
    const T* data() const { return data_; }

          T* begin()       { return data_; }
    const T* begin() const { return data_; }

          T* end()       { return data_ + size_; }
    const T* end() const { return data_ + size_; }

          T& operator[](std::size_t i)       { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    std::size_t size() const { return size_; }

    // Releases the ownership of the pointer to the caller.
    // After this call the array is in an empty state.
    // The returned pair is the pointer to the start of the array
    // and the logical number of elements (the allocation might be larger).
    std::pair<T*, std::size_t> release()
    {
        auto d = data_;
        auto s = size_;
        data_ = nullptr;
        size_ = 0;
        return { d, s };
    }

    // Changes the size of the array to a value smaller than the current size.
    // This only changes the logical size, no allocation nor copies take place.
    void truncate(std::size_t size)
    {
        assert(size <= size_);

        size_ = size;
    }

    ~DynArray()
    {
        if (data_ != nullptr)
            delete[] data_;
    }

private:
    T*          data_;
    std::size_t size_;
};

// This function takes a pointer `entries` to an array of entries of type
// `EntryType`, and the number of entires in the array `numEntries`, and
// creates a new array of symbols of type `SymbolType`.
// Each entry is interpreted as at least 1 symbol. The symbol size must
// evenly divide the entry size. If an entry is being interpreted as
// multiple symbols the symbols are ordered starting from lower bytes of the entry.
// For example let's consider EntryType == std::uint32_t and SymbolType == std::uint16_t:
// entry = 0x04030201
// symbols = [0x0201, 0x0403]
// Endianness is respected, but mixed endianness is not supported.
// Only little and big endian byte orders are supported.
template <typename EntryType, typename SymbolType>
DynArray<SymbolType> entries_to_symbols(const EntryType* entries, std::size_t numEntries)
{
    constexpr std::size_t entrySize       = sizeof(EntryType);
    constexpr std::size_t symbolSize      = sizeof(SymbolType);
    constexpr std::size_t symbolsPerEntry = entrySize / symbolSize;

    static_assert(entrySize % symbolSize == 0);
    static_assert(symbolSize == 1 || symbolSize == 2);
    static_assert(std::is_unsigned_v<SymbolType>);
    static_assert(std::is_integral_v<EntryType>);

    const std::size_t numSymbols = numEntries * entrySize / symbolSize;
    DynArray<SymbolType> symbols(numSymbols);

    if (IsLittleEndian || entrySize == symbolSize)
    {
        std::memcpy(&symbols[0], entries, numEntries * entrySize);
    }
    else
    {
        const char* rawData = reinterpret_cast<const char*>(entries);
        for (std::size_t i = 0; i < numEntries; ++i)
        {
            for (std::size_t j = 0; j < symbolsPerEntry; ++j)
            {
                std::memcpy(&symbols[i * symbolsPerEntry + (symbolsPerEntry - j - 1)],
                            rawData + i * entrySize + j * symbolSize,
                            symbolSize);
            }
        }
    }

    return symbols;
}

// Writes an integer into the `out` in little-endian byte order.
template <typename IntT>
void write_little_endian(std::uint8_t* out, IntT value)
{
    static_assert(std::is_integral_v<IntT>);
    static_assert(std::is_unsigned_v<IntT>);

    for (std::size_t i = 0; i < sizeof(IntT); ++i)
        out[i] = (value >> (i * 8)) & 0xFFu;
}

// Reads an integer from the `in` in little-endian byte order.
template <typename IntT>
IntT read_little_endian(const std::uint8_t* in)
{
    static_assert(std::is_integral_v<IntT>);
    static_assert(std::is_unsigned_v<IntT>);

    IntT v = 0;

    for (std::size_t i = 0; i < sizeof(IntT); ++i)
        v |= in[i] << (i * 8);

    return v;
}

template <typename SymbolT>
struct SymbolFrequencyTable
{
    using SymbolType    = SymbolT;
    using FrequencyType = std::uint32_t;
    using CountType     = std::uint32_t;

    static_assert(   std::is_same_v<SymbolT, std::uint8_t>
                  || std::is_same_v<SymbolT, std::uint16_t>);

    // The frequency type must be big enough to contain frequences for 2^16 symbols.
    // Also the size of it determines possible ranges for other parameters, and
    // how much space the dictionary requires. uint32_t is the sweet spot.
    static_assert(std::is_same_v<FrequencyType, std::uint32_t>);

    static constexpr std::size_t   symbolSize          = sizeof(SymbolType);
    static constexpr std::size_t   frequencySize       = sizeof(FrequencyType);
    static constexpr std::size_t   countSize           = sizeof(CountType);

    // The eof symbol does not have actual representation (so can't be stored in SymbolType).
    // It is the last symbol in the list.
    static constexpr std::uint64_t eofSymbol           = 0x10000;

    // The total frequency is made constant so that costly division can be replaced
    // with cheaper operations. This dramatically reduces the compression time.
    // The symbol frequency is scaled to match this total as close as possible.
    // The only available value given other constraints is 2^31-1.
    static constexpr std::uint32_t totalFrequency      = 0x7FFFFFFFu;
    static constexpr std::size_t   serializedEntrySize = symbolSize + frequencySize;

    // Individual frequencies for each symbol (excluding the eofSymbol).
    // Scaled such that the total is equal to `totalFrequency`
    FrequencyType frequencies          [eofSymbol + 1];

    // cumulativeFrequencies[i] = sum(frequencies[k < i])
    // cumulativeFrequencies[eofSymbol] = totalFrequency
    FrequencyType cumulativeFrequencies[eofSymbol + 2];

    // Creates an empty frequency table. The table is in undefined state
    // and invariants do not hold.
    SymbolFrequencyTable() {}

    // Resets the table to an undefined stated.
    void reset()
    {
        std::memset(frequencies,           0, (eofSymbol + 1) * frequencySize);
        std::memset(cumulativeFrequencies, 0, (eofSymbol + 2) * frequencySize);
    }

    // Initializes the table with frequencies extracted from the given `symbols` array.
    // The implementation requires that the number of symbols be less than `totalFrequency`.
    // This implementation detail can be improved in the future if required.
    // Returns true on success, otherwise false.
    bool for_symbols(const DynArray<SymbolType>& symbols)
    {
        reset();

        if (symbols.size() >= totalFrequency)
            return false;

        // The frequencies are scaled to have a total of `totalFrequency`.
        // +1 because we have to encode the eofSymbol too.
        const double mul = static_cast<double>(totalFrequency) / (symbols.size() + 1);
        double fractionalFrequencies[eofSymbol + 2] = {0.0};

        for (auto symbol : symbols)
            fractionalFrequencies[symbol] += mul;

        fractionalFrequencies[eofSymbol] = mul;

        FrequencyType sum = 0;
        for (std::size_t i = 0; i < eofSymbol; ++i)
        {
            // Perform rounding as quantization of the exact frequencies.
            // mul is guaranteed to be >= 1 so no frequency will be incorrectly truncated to 0.
            frequencies[i] = static_cast<FrequencyType>(fractionalFrequencies[i] + 0.5);
            sum += frequencies[i];
            cumulativeFrequencies[i + 1] = sum;
        }

        // Adjust the sum to be exactly `totalFrequency` in a best-effort way.
        cumulativeFrequencies[eofSymbol + 1] = totalFrequency;
        frequencies[eofSymbol] = totalFrequency - sum;

        return true;
    }

    // Serialization of the frequency table into a `out` byte stream. The memory pointed to
    // by `out` is assumed to have `capacity` number of bytes, and the function will fail if
    // reads go beyond that. The actual size of the serialized table is stored into
    // the `actualSize` out parameter, unless the function failed, in which case the
    // value in `actualSize` is undefined.
    // Returns true on success, otherwise false.
    //
    // The binary serialized format is as follows:
    //
    // Header:
    //   - num_used_symbols : uint32, little-endian
    //       the number of frequency values that are non-zero, excluding the eof symbol frequency
    //   - eof_frequency : uint32, little-endian
    //
    // Entry:
    //   - symbol : SymbolType, little-endian
    //   - frequency : uint32, little-endian
    //
    // Serialized:
    //   - header : Header
    //   - entries : Entry[header.num_used_symbols]
    //       entries corresponding to all symbols with non-zero frequency, excluding the eof symbol
    //
    bool serialize(std::uint8_t* out, std::size_t capacity, std::size_t& actualSize) const
    {
        const CountType numUsedSymbols = num_used_symbols();

        actualSize = serialized_size(numUsedSymbols);

        if (capacity < actualSize)
            return false;

        write_little_endian<CountType>    (out,             numUsedSymbols      );
        write_little_endian<FrequencyType>(out + countSize, frequencies[eofSymbol]);

        std::size_t numWrittenEntries = 0;
        for (std::size_t i = 0; i < eofSymbol; ++i)
        {
            const FrequencyType freq = frequencies[i];

            if (freq == 0)
                continue;

            const std::size_t offset =
                  countSize
                + frequencySize
                + numWrittenEntries * serializedEntrySize;

            write_little_endian<SymbolType>   (out + offset,              static_cast<SymbolType>(i));
            write_little_endian<FrequencyType>(out + offset + symbolSize, freq                      );

            numWrittenEntries += 1;
        }

        assert(numWrittenEntries == numUsedSymbols);

        return true;
    }

    // Deserializes a frequency table from the given `raw` byte stream.
    // The deserialization fails if the serialized data uses more than `size` bytes.
    // The actual size (number of bytes read) is saved in `actualSize`,
    // unless the function failed, in which case the value is left unspecified.
    // Returns true on success, otherwise false.
    bool deserialize(const std::uint8_t* raw, std::size_t size, std::size_t& actualSize)
    {
        const CountType numUsedSymbols = read_little_endian<CountType>(raw);

        actualSize = serialized_size(numUsedSymbols);

        if (actualSize > size)
            return false;

        reset();

        frequencies[eofSymbol] = read_little_endian<FrequencyType>(raw + countSize);

        for (std::size_t i = 0; i < numUsedSymbols; ++i)
        {
            const std::size_t   offset =
                  countSize
                + frequencySize
                + i * serializedEntrySize;

            const SymbolType    symbol = read_little_endian<SymbolType>   (raw + offset             );
            const FrequencyType freq   = read_little_endian<FrequencyType>(raw + offset + symbolSize);
            frequencies[symbol] = freq;
        }

        FrequencyType sum = 0;
        for (std::size_t i = 0; i < eofSymbol + 1; ++i)
        {
            sum += frequencies[i];
            cumulativeFrequencies[i + 1] = sum;
        }

        if (sum != totalFrequency)
            return false;

        return true;
    }

    // Returns the number of symbols with non-zero frequency, excluding the eof symbol.
    CountType num_used_symbols() const
    {
        CountType numUsedSymbols = 0;

        for (std::size_t i = 0; i < eofSymbol; ++i)
            numUsedSymbols += (frequencies[i] != 0);

        return numUsedSymbols;
    }

    // Returns the exact size of this table in the serialized form.
    std::size_t serialized_size() const
    {
        return serialized_size(num_used_symbols());
    }

private:
    // Returns the exact size of this table in the serialized form,
    // given the number of used symbols that need to be stored.
    std::size_t serialized_size(CountType numUsedSymbols) const
    {
        return
              countSize
            + frequencySize
            + numUsedSymbols * serializedEntrySize;
    }
};

namespace Detail
{
    struct OutputBitStream
    {
        // Create a bit writer, with the start byte being `values[0]` and maximum number of
        // writable bytes being `capacity`. The behaviour when Writing more than
        // `capacity * 8` bits is undefined.
        // The bits in each byte are populated starting from the least significant bit.
        // `capacity` number of bytes following the memory pointed to by `values`
        // is assumed to be zero initialized.
        OutputBitStream(std::uint8_t* values, std::size_t capacity) :
            values_(values),
            capacity_(capacity),
            numWrittenBits_(0)
        {
        }

        // Writes a single bit to the output. The value of `bit` must be either 0 or 1,
        // otherwise the behaviour is undefined.
        void write_bit(int bit)
        {
            assert(numWrittenBits_ / 8 < capacity_);
            assert(bit == 0 || bit == 1);

            values_[numWrittenBits_ / 8] |= bit << (numWrittenBits_ % 8);

            ++numWrittenBits_;
        }

        // Returns the number of bytes that had at least 1 bit written to them.
        std::size_t num_touched_bytes() const { return (numWrittenBits_ + 7) / 8; }

    private:
        std::uint8_t* values_;
        std::size_t   capacity_;
        std::size_t   numWrittenBits_;
    };

    struct InputBitStream
    {
        // Creates an bit reader, starting with the byte `data[0]` and containing `size * 8`
        // bits. Reading past `size * 8` bits is allowed and any such read will return 0.
        InputBitStream(const std::uint8_t* data, std::size_t size) :
            data_(data),
            size_(size * 8),
            numReadBits_(0)
        {
        }

        // Reads the next bit and advances the head. The returned value is either 0 or 1.
        int read_bit()
        {
            const int bit = peek_bit();

            ++numReadBits_;

            return bit;
        }

        // Reads the next bit without advancing the head. The returned value is either 0 or 1.
        int peek_bit() const
        {
            const std::size_t byte = numReadBits_ / 8;

            const int bit =
                numReadBits_ < size_
                ? (data_[byte] >> (numReadBits_ % 8))
                : 0;

            return bit & 1;
        }

    private:
        const std::uint8_t* data_;
              std::size_t   size_;
              std::size_t   numReadBits_;
    };

    // Parameters governing the size of the state used to keep track of the current
    // position in the "infinite" fraction being coded. Given the choice of FrequencyType
    // and the `totalFrequency` these are pretty much the only allowed parameter choices.
    static constexpr std::uint64_t numStateBits   = 33;
    static constexpr std::uint64_t fullRange      = 1ull << numStateBits;
    static constexpr std::uint64_t stateMask      = fullRange - 1;
    static constexpr std::uint64_t halfRange      = fullRange >> 1;
    static constexpr std::uint64_t quarterRange   = halfRange >> 1;
    static constexpr std::uint64_t minimumRange   = quarterRange + 2;
    static constexpr std::uint64_t maximumTotal   =
        std::min(std::numeric_limits<decltype(fullRange)>::max() / fullRange,
                 minimumRange);

    // This class encodes a single array of values passed in the constructor.
    // It's a class only because it makes it easy to share state between functions.
    template <typename EntryT, typename SymbolT>
    struct Encoder
    {
        using EntryType                = EntryT;
        using SymbolType               = SymbolT;
        using SymbolFrequencyTableType = SymbolFrequencyTable<SymbolType>;
        using FrequencyType            = typename SymbolFrequencyTableType::FrequencyType;

        static constexpr std::size_t entrySize  = sizeof(EntryType);
        static constexpr std::size_t symbolSize = sizeof(SymbolType);

        static_assert(SymbolFrequencyTableType::totalFrequency <= maximumTotal);
        static_assert(std::is_unsigned_v<SymbolType>);
        static_assert(std::is_integral_v<SymbolType>);
        static_assert(entrySize % symbolSize == 0);
        static_assert(symbolSize == 1 || symbolSize == 2);

        // Encodes the `numEntries` entries of type `EntryType`, decomposed into
        // parts of `SymbolType` (as described in `entries_to_symbols(...)`), into a bit stream
        // using arithmetic coding with a fixed frequency table.
        // The operation may fail, in which case `is_good()` will return false.
        // To retrieve the results call `result()` on an rvalue.
        // All work is performed in the constructor.
        // The format of the output is as follows:
        //   - frequency_table : see SymbolFrequencyTable::serialize(...)
        //   - encoded_bit_stream : uint8[...]
        //       The size is not encoded. The stream ends with an encoded eof symbol.
        Encoder(const EntryType* entries, std::size_t numEntries)
        {
            symbols_          = entries_to_symbols<EntryType, SymbolType>(entries, numEntries);
            good_             = frequencyTable_.for_symbols(symbols_);
            if (!good_)
                return;

            uncompressedSize_ = numEntries * entrySize;
            high_             = stateMask;
            low_              = 0;
            oppositeBits_     = 0;

            good_             = encode();
        }

        // Returns whether the encoding succeeded.
        bool is_good() const { return good_; }

        // Returns the resulting encoded bytes and whether the encoding succeeded.
        auto result() && { return std::make_pair(std::move(compressed_), good_); }

    private:
        DynArray<SymbolType>     symbols_;
        SymbolFrequencyTableType frequencyTable_;
        std::size_t              uncompressedSize_;
        std::uint64_t            high_;
        std::uint64_t            low_;
        std::uint64_t            oppositeBits_;
        DynArray<std::uint8_t>   compressed_;
        bool                     good_;

        bool encode()
        {
            const std::size_t freqTableSize = frequencyTable_.serialized_size();
            const std::size_t safetyMargin = 256;
            const std::size_t totalSize =
                  freqTableSize
                + uncompressedSize_
                + safetyMargin;

            compressed_ = DynArray<std::uint8_t>(totalSize);

            std::size_t actualFreqTableSize;
            if (   !frequencyTable_.serialize(compressed_.data(),
                                              compressed_.size(),
                                              actualFreqTableSize)
                || freqTableSize != actualFreqTableSize)
            {
                return false;
            }

            OutputBitStream out(&compressed_[freqTableSize], uncompressedSize_);
            for (auto symbol : symbols_)
            {
                encode_symbol(out, symbol);
                if (out.num_touched_bytes() > uncompressedSize_)
                    return false;
            }

            encode_end(out);
            if (out.num_touched_bytes() > uncompressedSize_)
                return false;

            const std::size_t actualTotalSize = freqTableSize + out.num_touched_bytes();
            compressed_.truncate(actualTotalSize);

            return true;
        }

        void encode_end(OutputBitStream& out)
        {
            encode_symbol(out, SymbolFrequencyTableType::eofSymbol);
            out.write_bit(1);
        }

        void encode_symbol(OutputBitStream& out, std::uint64_t symbol)
        {
            assert(low_  <  high_);
            assert(low_  <  fullRange);
            assert(high_ <  fullRange);
            assert(low_  <  halfRange);
            assert(high_ >= halfRange);
            assert(low_  <  quarterRange
                || high_ >= quarterRange * 3);

            const std::uint64_t range = high_ - low_ + 1;
            assert(range >= minimumRange);
            assert(range <= fullRange);

            const FrequencyType symLow  = frequencyTable_.cumulativeFrequencies[symbol    ];
            const FrequencyType symHigh = frequencyTable_.cumulativeFrequencies[symbol + 1];
            assert(symLow  != symHigh);
            assert(symHigh <= SymbolFrequencyTableType::totalFrequency);

            high_ = low_ + (range * symHigh) / SymbolFrequencyTableType::totalFrequency - 1;
            low_  = low_ + (range * symLow ) / SymbolFrequencyTableType::totalFrequency    ;

            assert(low_  < high_);
            assert(low_  < fullRange);
            assert(high_ < fullRange);

            while (((low_ ^ high_) & halfRange) == 0)
            {
                const int bit = (low_ >> (numStateBits - 1)) & 1;
                output_bits(out, bit);
                low_  =  low_  << 1;
                high_ = (high_ << 1) | 1;
            }

            assert(low_ < high_);
            assert((low_  & halfRange) == 0);
            assert((high_ & halfRange) == halfRange);

            while (((low_ & ~high_) & quarterRange) != 0)
            {
                oppositeBits_ += 1;
                low_  = (low_  << 1) ^ halfRange;
                high_ = (high_ << 1) ^ ((halfRange << 1) | halfRange | 1);
            }

            low_ &= stateMask;
            high_ &= stateMask;

            assert(low_  <  high_);
            assert(low_  <  fullRange);
            assert(high_ <  fullRange);
            assert(low_  <  halfRange);
            assert(high_ >= halfRange);
            assert(low_  <  quarterRange
                || high_ >= quarterRange * 3);
        }

        void output_bits(OutputBitStream& out, int bit)
        {
            out.write_bit(bit);
            while (oppositeBits_ > 0)
            {
                out.write_bit(bit ^ 1);
                oppositeBits_ -= 1;
            }
        }
    };

    // This class encodes a byte array previous encoded by the Encoder.
    // It's a class only because it makes it easy to share state between functions.
    template <typename EntryT, typename SymbolT>
    struct Decoder
    {
        using EntryType                = EntryT;
        using SymbolType               = SymbolT;
        using SymbolFrequencyTableType = SymbolFrequencyTable<SymbolType>;
        using FrequencyType            = typename SymbolFrequencyTableType::FrequencyType;

        static constexpr std::size_t entrySize  = sizeof(EntryType);
        static constexpr std::size_t symbolSize = sizeof(SymbolType);
        static constexpr std::size_t symbolsPerEntry = entrySize / symbolSize;

        static_assert(SymbolFrequencyTableType::totalFrequency <= maximumTotal);
        static_assert(std::is_unsigned_v<SymbolType>);
        static_assert(std::is_integral_v<SymbolType>);
        static_assert(entrySize % symbolSize == 0);
        static_assert(symbolSize == 1 || symbolSize == 2);

        // Decodes the data pointed to by `compressed`. The data is assumed to have size
        // `compressedSize` and containing `numEntries` entries.
        // The operation may fail, in which case `is_good()` will return false.
        // To retrieve the results call `result()` on an rvalue.
        // All work is performed in the constructor.
        Decoder(const std::uint8_t* compressed, std::size_t compressedSize, std::size_t numEntries)
        {
            compressed_     = compressed;
            compressedSize_ = compressedSize;
            good_           = frequencyTable_.deserialize(compressed,
                                                          compressedSize,
                                                          frequencyTableSerializedSize_);
            if (!good_)
                return;

            numEntries_     = numEntries;
            high_           = stateMask;
            low_            = 0;
            code_           = 0;

            good_           = decode();
        }

        // Returns whether the encoding succeeded.
        bool is_good() const { return good_; }

        // Returns the resulting encoded bytes and whether the encoding succeeded.
        auto result() && { return std::make_pair(std::move(entries_), good_); }

    private:
        const std::uint8_t*      compressed_;
        std::size_t              compressedSize_;
        SymbolFrequencyTableType frequencyTable_;
        std::size_t              frequencyTableSerializedSize_;
        std::size_t              numEntries_;
        std::uint64_t            high_;
        std::uint64_t            low_;
        std::uint64_t            code_;
        DynArray<EntryType>      entries_;
        bool                     good_;

        bool decode()
        {
            InputBitStream in(compressed_ + frequencyTableSerializedSize_,
                              compressedSize_ - frequencyTableSerializedSize_);

            for (std::size_t i = 0; i < numStateBits; ++i)
            {
                code_ <<= 1;
                code_ |= in.read_bit();
            }

            entries_ = DynArray<EntryType>(numEntries_);

            char* rawData = reinterpret_cast<char*>(entries_.data());
            for (std::size_t i = 0; i < numEntries_; ++i)
            {
                for (std::size_t j = 0; j < symbolsPerEntry; ++j)
                {
                    const FrequencyType value = next_value();

                    if (is_decoded_value_eof_symbol(value))
                        return false;

                    SymbolType symbol = decode_value(value);
                    if (frequencyTable_.frequencies[symbol] == 0)
                        return false;

                    update_on_symbol(in, symbol);

                    const std::size_t destinationOffset =
                        (IsLittleEndian || entrySize == symbolSize)
                        ? (i * entrySize +                    j      * symbolSize)
                        : (i * entrySize + (symbolsPerEntry - j - 1) * symbolSize);

                    std::memcpy(rawData + destinationOffset,
                                &symbol,
                                symbolSize);
                }
            }

            if (!is_decoded_value_eof_symbol(next_value()))
                return false;

            return true;
        }

        FrequencyType next_value() const
        {
            const std::uint64_t range  = high_ - low_ + 1;
            const std::uint64_t offset = code_ - low_;
            const FrequencyType value  =
                static_cast<FrequencyType>(
                    ((offset + 1) * SymbolFrequencyTableType::totalFrequency - 1) / range);

            assert(value * range / SymbolFrequencyTableType::totalFrequency <= offset);
            assert(value < SymbolFrequencyTableType::totalFrequency);

            return value;
        }

        void assert_next_symbol_correct(SymbolType symbol) const
        {
#if !defined (NDEBUG)
            const std::uint64_t range  = high_ - low_ + 1;
            const std::uint64_t offset = code_ - low_;

            assert(   offset >= frequencyTable_.cumulativeFrequencies[symbol    ] * range
                                / SymbolFrequencyTableType::totalFrequency
                   || offset <  frequencyTable_.cumulativeFrequencies[symbol + 1] * range
                                / SymbolFrequencyTableType::totalFrequency);
#endif
            (void)symbol;
        }

        bool is_decoded_value_eof_symbol(FrequencyType value) const
        {
            return value > frequencyTable_.cumulativeFrequencies[SymbolFrequencyTableType::eofSymbol];
        }

        SymbolType decode_value(FrequencyType value) const
        {
            assert(!is_decoded_value_eof_symbol(value));

            // Assumes the symbol is not SymbolFrequencyTableType::eofSymbol,
            // because then the search space is of a power of 2 size.
            static_assert(   SymbolFrequencyTableType::eofSymbol == 0x10000
                          || SymbolFrequencyTableType::eofSymbol == 0x00100);

            SymbolType symbol = 0;
            if constexpr (SymbolFrequencyTableType::eofSymbol == 0x10000)
            {
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 << 15)] <= value) << 15;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 << 14)] <= value) << 14;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 << 13)] <= value) << 13;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 << 12)] <= value) << 12;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 << 11)] <= value) << 11;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 << 10)] <= value) << 10;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  9)] <= value) <<  9;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  8)] <= value) <<  8;
            }
            {
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  7)] <= value) <<  7;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  6)] <= value) <<  6;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  5)] <= value) <<  5;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  4)] <= value) <<  4;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  3)] <= value) <<  3;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  2)] <= value) <<  2;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  1)] <= value) <<  1;
                symbol += (frequencyTable_.cumulativeFrequencies[symbol + (1 <<  0)] <= value) <<  0;
            }

            assert_next_symbol_correct(symbol);

            return symbol;
        }

        void update_on_symbol(InputBitStream& in, SymbolType symbol)
        {
            assert(low_  <  high_);
            assert(low_  <  fullRange);
            assert(high_ <  fullRange);
            assert(low_  <  halfRange);
            assert(high_ >= halfRange);
            assert(low_  <  quarterRange
                || high_ >= quarterRange * 3);

            const std::uint64_t range = high_ - low_ + 1;
            assert(range >= minimumRange);
            assert(range <= fullRange);

            const std::uint32_t symLow  = frequencyTable_.cumulativeFrequencies[symbol    ];
            const std::uint32_t symHigh = frequencyTable_.cumulativeFrequencies[symbol + 1];
            assert(symLow != symHigh);
            assert(symHigh <= frequencyTable_.totalFrequency);

            high_ = low_ + (range * symHigh) / frequencyTable_.totalFrequency - 1;
            low_  = low_ + (range * symLow ) / frequencyTable_.totalFrequency;

            assert(low_ < high_);
            assert(low_ < fullRange);
            assert(high_ < fullRange);

            while (((low_ ^ high_) & halfRange) == 0)
            {
                code_ = (code_ << 1) | in.read_bit();
                low_  =  low_  << 1;
                high_ = (high_ << 1) | 1;
            }

            assert(low_ < high_);
            assert((low_  & halfRange) == 0);
            assert((high_ & halfRange) == halfRange);

            std::uint64_t codeHalfRangeBit = code_ & halfRange;
            while (((low_ & ~high_) & quarterRange) != 0)
            {
                code_ = (code_ << 1) ^ in.read_bit();
                low_  = (low_  << 1) ^ halfRange;
                high_ = (high_ << 1) ^ ((halfRange << 1) | halfRange | 1);
            }

            code_ = (code_ & (stateMask >> 1)) | codeHalfRangeBit;
            low_ &= stateMask;
            high_ &= stateMask;

            assert(low_  <  high_);
            assert(low_  <  fullRange);
            assert(high_ <  fullRange);
            assert(low_  <  halfRange);
            assert(high_ >= halfRange);
            assert(low_  <  quarterRange
                || high_ >= quarterRange * 3);
            assert(code_ >= low_
                && code_ <= high_);
        }
    };
}

template <typename EntryT, typename SymbolT>
std::pair<DynArray<std::uint8_t>, bool> compress(
    const EntryT* entries,
    std::size_t numEntries)
{
    Detail::Encoder<EntryT, SymbolT> enc(entries, numEntries);
    return std::move(enc).result();
}

template <typename EntryT, typename SymbolT>
std::pair<DynArray<EntryT>, bool> decompress(
    const std::uint8_t* compressed,
    std::size_t compressedSize,
    std::size_t numEntries)
{
    Detail::Decoder<EntryT, SymbolT> dec(compressed, compressedSize, numEntries);
    return std::move(dec).result();
}

extern "C"
{
    struct CompressedBlock
    {
        std::uint8_t* data;
        std::size_t   size;
        std::size_t   numEntries;

        CompressedBlock(std::pair<DynArray<std::uint8_t>, bool> result, std::size_t ne)
        {
            auto&& [compressed, good] = result;
            if (good)
            {
                auto d = compressed.release();
                data = d.first;
                size = d.second;
                numEntries = ne;
            }
            else
            {
                data = nullptr;
                size = 0;
                numEntries = 0;
            }
        }
    };

    EXPORT void CDECL ac_destroy_compressed_block(CompressedBlock* block)
    {
        delete block;
    }
}

namespace Detail
{
    template <typename EntryType, typename SymbolType>
    CompressedBlock* generic_compress_c_api(EntryType* entries, std::size_t numEntries)
    {
        return new CompressedBlock(compress<EntryType, SymbolType>(entries, numEntries),
                                   numEntries);
    }

    template <typename EntryType, typename SymbolType>
    EntryType* generic_decompress_c_api(CompressedBlock* compressed)
    {
        if (compressed->data == nullptr)
            return nullptr;

        auto [decompressed, good] =
            decompress<EntryType, SymbolType>(compressed->data,
                                              compressed->size,
                                              compressed->numEntries);
        return good ? decompressed.release().first : nullptr;
    }
}

extern "C"
{
    // COMPRESSION:

    //     8 bit entries

    EXPORT CompressedBlock* CDECL ac_compress_u8_entry_u8_symbol(std::uint8_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::uint8_t, std::uint8_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_i8_entry_u8_symbol(std::int8_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::int8_t, std::uint8_t>(entries, numEntries);
    }

    //     16 bit entries

    EXPORT CompressedBlock* CDECL ac_compress_u16_entry_u8_symbol(std::uint16_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::uint16_t, std::uint8_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_u16_entry_u16_symbol(std::uint16_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::uint16_t, std::uint16_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_i16_entry_u8_symbol(std::int16_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::int16_t, std::uint8_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_i16_entry_u16_symbol(std::int16_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::int16_t, std::uint16_t>(entries, numEntries);
    }

    //     32 bit entries

    EXPORT CompressedBlock* CDECL ac_compress_u32_entry_u8_symbol(std::uint32_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::uint32_t, std::uint8_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_u32_entry_u16_symbol(std::uint32_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::uint32_t, std::uint16_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_i32_entry_u8_symbol(std::int32_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::int32_t, std::uint8_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_i32_entry_u16_symbol(std::int32_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::int32_t, std::uint16_t>(entries, numEntries);
    }

    //     64 bit entries

    EXPORT CompressedBlock* CDECL ac_compress_u64_entry_u8_symbol(std::uint64_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::uint64_t, std::uint8_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_u64_entry_u16_symbol(std::uint64_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::uint64_t, std::uint16_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_i64_entry_u8_symbol(std::int64_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::int64_t, std::uint8_t>(entries, numEntries);
    }

    EXPORT CompressedBlock* CDECL ac_compress_i64_entry_u16_symbol(std::int64_t* entries, std::size_t numEntries)
    {
        return Detail::generic_compress_c_api<std::int64_t, std::uint16_t>(entries, numEntries);
    }

    // DECOMPRESSION:

    //     8 bit entries

    EXPORT std::uint8_t* CDECL ac_decompress_u8_entry_u8_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::uint8_t, std::uint8_t>(compressed);
    }

    EXPORT std::int8_t* CDECL ac_decompress_i8_entry_u8_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::int8_t, std::uint8_t>(compressed);
    }

    //     16 bit entries

    EXPORT std::uint16_t* CDECL ac_decompress_u16_entry_u8_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::uint16_t, std::uint8_t>(compressed);
    }

    EXPORT std::uint16_t* CDECL ac_decompress_u16_entry_u16_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::uint16_t, std::uint16_t>(compressed);
    }

    EXPORT std::int16_t* CDECL ac_decompress_i16_entry_u8_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::int16_t, std::uint8_t>(compressed);
    }

    EXPORT std::int16_t* CDECL ac_decompress_i16_entry_u16_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::int16_t, std::uint16_t>(compressed);
    }

    //     32 bit entries

    EXPORT std::uint32_t* CDECL ac_decompress_u32_entry_u8_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::uint32_t, std::uint8_t>(compressed);
    }

    EXPORT std::uint32_t* CDECL ac_decompress_u32_entry_u16_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::uint32_t, std::uint16_t>(compressed);
    }

    EXPORT std::int32_t* CDECL ac_decompress_i32_entry_u8_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::int32_t, std::uint8_t>(compressed);
    }

    EXPORT std::int32_t* CDECL ac_decompress_i32_entry_u16_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::int32_t, std::uint16_t>(compressed);
    }

    //     64 bit entries

    EXPORT std::uint64_t* CDECL ac_decompress_u64_entry_u8_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::uint64_t, std::uint8_t>(compressed);
    }

    EXPORT std::uint64_t* CDECL ac_decompress_u64_entry_u16_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::uint64_t, std::uint16_t>(compressed);
    }

    EXPORT std::int64_t* CDECL ac_decompress_i64_entry_u8_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::int64_t, std::uint8_t>(compressed);
    }

    EXPORT std::int64_t* CDECL ac_decompress_i64_entry_u16_symbol(CompressedBlock* compressed)
    {
        return Detail::generic_decompress_c_api<std::int64_t, std::uint16_t>(compressed);
    }

    // DELETION:

    EXPORT void CDECL ac_destroy_entries_u8(std::uint8_t* ptr)
    {
        delete ptr;
    }

    EXPORT void CDECL ac_destroy_entries_u16(std::uint16_t* ptr)
    {
        delete ptr;
    }

    EXPORT void CDECL ac_destroy_entries_u32(std::uint32_t* ptr)
    {
        delete ptr;
    }

    EXPORT void CDECL ac_destroy_entries_u64(std::uint64_t* ptr)
    {
        delete ptr;
    }

    EXPORT void CDECL ac_destroy_entries_i8(std::int8_t* ptr)
    {
        delete ptr;
    }

    EXPORT void CDECL ac_destroy_entries_i16(std::int16_t* ptr)
    {
        delete ptr;
    }

    EXPORT void CDECL ac_destroy_entries_i32(std::int32_t* ptr)
    {
        delete ptr;
    }

    EXPORT void CDECL ac_destroy_entries_i64(std::int64_t* ptr)
    {
        delete ptr;
    }
}
