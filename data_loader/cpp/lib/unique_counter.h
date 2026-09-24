#pragma once

// UniquePositionCounter: counts approximate unique positions, total
// positions (after filtering), and preskip positions (before filtering)
// seen by the data loader, using HyperLogLog.
//
// The HLL registers are updated via std::atomic CAS-max; stats() and
// serialize() read them with relaxed atomic loads.
//
// Initial-state restore for checkpoint restart is race-free: the
// registers are populated in the constructor, before any worker
// threads start.

#include "hyperloglog.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace nnue {

class UniquePositionCounter {
public:
    UniquePositionCounter() :
        m_registers(std::make_unique<std::atomic<std::uint8_t>[]>(hll::HyperLogLog::kNumRegisters)),
        m_total(0),
        m_preskip(0)
    {
        for (std::size_t i = 0; i < hll::HyperLogLog::kNumRegisters; ++i)
            m_registers[i].store(0, std::memory_order_relaxed);
    }

    UniquePositionCounter(const std::uint8_t* initial_hll, std::size_t initial_hll_size, std::uint64_t initial_total, std::uint64_t initial_preskip = 0) :
        UniquePositionCounter()
    {
        if (initial_hll && initial_hll_size >= hll::HyperLogLog::kSerializedSize)
        {
            auto h = hll::HyperLogLog::deserialize(initial_hll, initial_hll_size);
            const auto& regs = h.registers();
            for (std::size_t i = 0; i < regs.size(); ++i)
                m_registers[i].store(regs[i], std::memory_order_relaxed);
        }
        m_total.store(initial_total, std::memory_order_relaxed);
        m_preskip.store(initial_preskip, std::memory_order_relaxed);
    }

    void addBatch(std::vector<std::uint64_t>&& keys) noexcept
    {
        for (const auto key : keys)
        {
            const std::size_t j = key >> (64 - hll::HyperLogLog::kBitsUsed);
            const std::uint64_t w = key << hll::HyperLogLog::kBitsUsed;
            const std::uint8_t rank = static_cast<std::uint8_t>(std::countr_zero(w));

            std::uint8_t old = m_registers[j].load(std::memory_order_relaxed);
            while (rank > old &&
                   !m_registers[j].compare_exchange_weak(
                       old, rank, std::memory_order_relaxed, std::memory_order_relaxed))
            {
            }
        }
        m_total.fetch_add(keys.size(), std::memory_order_relaxed);
    }

    void add(std::uint64_t key) noexcept
    {
        addBatch(std::vector<std::uint64_t>{key});
    }

    void addPreskip(std::uint64_t n) noexcept
    {
        m_preskip.fetch_add(n, std::memory_order_relaxed);
    }

    void stats(std::uint64_t& preskip, std::uint64_t& total, std::uint64_t& unique) const noexcept
    {
        preskip = m_preskip.load(std::memory_order_relaxed);
        total = m_total.load(std::memory_order_relaxed);
        hll::HyperLogLog snapshot;
        auto& regs = const_cast<std::vector<std::uint8_t>&>(snapshot.registers());
        for (std::size_t i = 0; i < hll::HyperLogLog::kNumRegisters; ++i)
            regs[i] = m_registers[i].load(std::memory_order_relaxed);
        unique = snapshot.count();
    }

    std::vector<std::uint8_t> serialize() const
    {
        hll::HyperLogLog snapshot;
        auto& regs = const_cast<std::vector<std::uint8_t>&>(snapshot.registers());
        for (std::size_t i = 0; i < hll::HyperLogLog::kNumRegisters; ++i)
            regs[i] = m_registers[i].load(std::memory_order_relaxed);
        return snapshot.serialize();
    }

private:
    std::unique_ptr<std::atomic<std::uint8_t>[]> m_registers;
    std::atomic<std::uint64_t> m_total;
    std::atomic<std::uint64_t> m_preskip;
};

} // namespace nnue
