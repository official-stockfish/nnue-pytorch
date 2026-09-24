#pragma once

// 64-bit hash of chess::Position for use in HyperLogLog.

#include <cstdint>
#include <cstring>

#include "chess.h"

namespace nnue::hash {

[[nodiscard]] inline std::uint64_t hash(const chess::Position& pos) noexcept
{
    const auto c = pos.compress();

    static_assert(sizeof(chess::CompressedPosition) == 24);
    static_assert(std::is_trivially_copyable_v<chess::CompressedPosition>);

    std::uint64_t w[3];
    std::memcpy(w, &c, 24);

    std::uint64_t h = 0;
    for (int i = 0; i < 3; ++i)
    {
        h ^= w[i];
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebULL;
        h ^= h >> 31;
    }
    return h;
}

} // namespace nnue::hash
