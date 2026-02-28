#pragma once

#include <random>

namespace rng
{
    template <typename RNG = std::mt19937_64>
    auto& get_thread_local_rng(typename RNG::result_type seed = RNG::default_seed)
    {
        static thread_local RNG s_rng(seed);
        return s_rng;
    }
}
