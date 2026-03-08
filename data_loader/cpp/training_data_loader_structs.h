#pragma once

struct DataloaderSkipConfig {
    bool   filtered;
    int    random_fen_skipping;
    bool   wld_filtered;
    int    early_fen_skipping;
    int    simple_eval_skipping;
    int    param_index;
    double pc_y1, pc_y2, pc_y3;
};

struct DataloaderDDPConfig {
    int rank;
    int world_size;
};
