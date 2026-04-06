#pragma once

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
