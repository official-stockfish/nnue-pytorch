#pragma once

// HyperLogLog cardinality estimator — faithful C++ port of
// chess-binpack-utils/src/unique.rs (BITSUSED=20, 2^20 u8 registers,
// ~1 MiB, ~0.1% standard error).
//
// Reference paper: Heule, Nunkesser, Hall "HyperLogLog in Practice"
// https://oertl.github.io/hyperloglog-sketch-estimation-paper/paper.pdf

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nnue::hll {

class HyperLogLog {
public:
    static constexpr unsigned kBitsUsed = 20;
    static constexpr std::size_t kNumRegisters = std::size_t{1} << kBitsUsed;
    static constexpr std::size_t kSerializedSize = 16 + kNumRegisters; // header + registers

    HyperLogLog() : m_registers(kNumRegisters, 0) {}

    explicit HyperLogLog(std::vector<uint8_t> regs)
        : m_registers(std::move(regs))
    {
        if (m_registers.size() != kNumRegisters)
            m_registers.resize(kNumRegisters, 0);
    }

    void add(std::uint64_t data) noexcept
    {
        const std::size_t j = data >> (64 - kBitsUsed);
        const std::uint64_t w = data << kBitsUsed;
        const std::uint8_t rank = static_cast<std::uint8_t>(std::countr_zero(w));
        if (rank > m_registers[j])
            m_registers[j] = rank;
    }

    [[nodiscard]] std::uint64_t count() const noexcept
    {
        const double size = static_cast<double>(kNumRegisters);
        // c[k] = number of registers with value k (after shifting by BITSUSED)
        std::array<std::uint32_t, 64 - kBitsUsed + 2> c{};
        for (const auto x : m_registers)
        {
            const int k = std::max(0, static_cast<int>(x) - static_cast<int>(kBitsUsed) + 1);
            c[static_cast<std::size_t>(k)] += 1;
        }

        double z = size * tau(1.0 - c[64 - kBitsUsed + 1] / size);
        for (int k = 64 - static_cast<int>(kBitsUsed); k >= 1; --k)
            z = 0.5 * (z + c[static_cast<std::size_t>(k)]);
        z += size * sigma(c[0] / size);

        return static_cast<std::uint64_t>(1.0 / (2.0 * std::log(2.0)) * size * size / z);
    }

    [[nodiscard]] std::vector<uint8_t> serialize() const
    {
        std::vector<uint8_t> out;
        out.reserve(kSerializedSize);
        // 16-byte header: "HLL1" + version + precision + 2 reserved + register count
        out.push_back('H'); out.push_back('L'); out.push_back('L'); out.push_back('1');
        out.push_back(0x01);                                   // version
        out.push_back(static_cast<uint8_t>(kBitsUsed));         // precision
        out.push_back(0x00); out.push_back(0x00);              // reserved
        const uint64_t cnt = kNumRegisters;
        for (int i = 0; i < 8; ++i)
            out.push_back(static_cast<uint8_t>((cnt >> (i * 8)) & 0xFF));
        out.insert(out.end(), m_registers.begin(), m_registers.end());
        return out;
    }

    static HyperLogLog deserialize(const uint8_t* data, std::size_t size)
    {
        if (size < 16 || std::memcmp(data, "HLL1", 4) != 0)
            throw std::runtime_error("Invalid HLL magic");
        const uint8_t version = data[4];
        if (version != 0x01)
            throw std::runtime_error("Unsupported HLL version");
        const uint8_t precision = data[5];
        if (precision != kBitsUsed)
            throw std::runtime_error("HLL precision mismatch");
        std::uint64_t cnt = 0;
        for (int i = 0; i < 8; ++i)
            cnt |= static_cast<uint64_t>(data[8 + i]) << (i * 8);
        if (cnt != kNumRegisters)
            throw std::runtime_error("HLL register count mismatch");
        if (size < kSerializedSize)
            throw std::runtime_error("HLL data too short");

        std::vector<uint8_t> regs(data + 16, data + 16 + kNumRegisters);
        return HyperLogLog(std::move(regs));
    }

    [[nodiscard]] const std::vector<uint8_t>& registers() const noexcept { return m_registers; }

private:
    static double tau(double x)
    {
        if (x == 0.0 || x == 1.0) return 0.0;
        double y = 1.0;
        double z = 1.0 - x;
        double zp;
        do {
            x = std::sqrt(x);
            zp = z;
            y /= 2.0;
            z -= (1.0 - x) * (1.0 - x) * y;
        } while (zp != z);
        return z / 3.0;
    }

    static double sigma(double x)
    {
        if (x == 1.0) return std::numeric_limits<double>::infinity();
        double y = 1.0;
        double z = x;
        double zp;
        do {
            x *= x;
            zp = z;
            z += x * y;
            y *= 2.0;
        } while (zp != z);
        return z;
    }

    std::vector<uint8_t> m_registers;
};

} // namespace nnue::hll
