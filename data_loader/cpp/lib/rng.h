#pragma once

#include <random>
#include <vector>
#include <algorithm>
#include <iterator>
#include <limits>

namespace rng
{
    // A standard library 64-bit linear congruential generator (LCG)
    // using Knuth's MMIX parameters.
    using lcg64 = std::linear_congruential_engine<std::uint64_t, 6364136223846793005ULL, 1442695040888963407ULL, 0ULL>;

    template <typename RNG = lcg64>
    auto& get_thread_local_rng(typename RNG::result_type seed = RNG::default_seed)
    {
        static thread_local RNG s_rng(seed);
        return s_rng;
    }

    template <typename IntType = std::size_t>
    class lightweight_discrete_distribution {
    public:
        lightweight_discrete_distribution() = default;

        template <typename InputIt>
        lightweight_discrete_distribution(InputIt first, InputIt last) {
            std::uint64_t sum = 0;
            for (auto it = first; it != last; ++it) {
                sum += static_cast<std::uint64_t>(*it);
                m_prefix_sums.push_back(sum);
            }
            m_total_weight = sum;
        }

        template <typename Generator>
        IntType operator()(Generator& g) const {
            if (m_prefix_sums.empty() || m_total_weight == 0) {
                return 0;
            }
            using result_type = typename Generator::result_type;
            constexpr result_type g_min = Generator::min();
            constexpr result_type g_max = Generator::max();
            constexpr std::uint64_t g_range = static_cast<std::uint64_t>(g_max) - static_cast<std::uint64_t>(g_min);

            std::uint64_t r = static_cast<std::uint64_t>(g()) - static_cast<std::uint64_t>(g_min);
            std::uint64_t val;
            if constexpr (g_range == std::numeric_limits<std::uint64_t>::max()) {
                val = static_cast<std::uint64_t>(
                    (static_cast<__uint128_t>(r) * m_total_weight) >> 64
                );
            } else {
                val = static_cast<std::uint64_t>(
                    (static_cast<__uint128_t>(r) * m_total_weight) / (g_range + 1)
                );
            }

            auto it = std::upper_bound(m_prefix_sums.begin(), m_prefix_sums.end(), val);
            if (it == m_prefix_sums.end()) {
                return m_prefix_sums.size() - 1;
            }
            return std::distance(m_prefix_sums.begin(), it);
        }

    private:
        std::vector<std::uint64_t> m_prefix_sums;
        std::uint64_t m_total_weight = 0;
    };
}
