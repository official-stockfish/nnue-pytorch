#pragma once

#include <random>

namespace rng
{
    inline auto& get_thread_local_rng()
    {
        static thread_local std::mt19937_64 s_rng(std::random_device{}());
        return s_rng;
    }
}
