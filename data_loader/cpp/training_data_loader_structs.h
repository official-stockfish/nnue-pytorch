#pragma once

#include <cstddef>
#include <cstdint>

struct DataloaderSkipConfig {
    bool   filtered;
    int    random_fen_skipping;
    bool   wld_filtered;
    int    early_fen_skipping;
    int    soft_early_fen_skipping;
    int    simple_eval_skipping;
    int    param_index;
    double pc_y0, pc_y1, pc_y2, pc_y3, pc_y4;
    double ply_x1, ply_y1, ply_x2, ply_y2,
           ply_x3, ply_y3, ply_x4, ply_y4;
};

struct DataloaderDDPConfig {
    int rank;
    int world_size;
};

struct DataloaderHllConfig {
    const std::uint8_t* initial_hll;      // may be nullptr
    std::size_t        initial_hll_size;  // bytes, 0 if no initial state
    std::uint64_t      initial_total;     // total count at restart
    std::uint64_t      initial_preskip;   // preskip count at restart
};
