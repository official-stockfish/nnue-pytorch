#pragma once

#include <cstddef>


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

// Opaque handles
using SparseBatch       = void;
using FenBatch          = void;
using FenBatchStream    = void;
using SparseBatchStream = void;

extern "C" {

// batches
SparseBatch* get_sparse_batch_from_fens(const char*        feature_set_c,
                                        int                num_fens,
                                        const char* const* fens,
                                        int*               scores,
                                        int*               plies,
                                        int*               results);

void destroy_sparse_batch(SparseBatch* e);
void destroy_fen_batch(FenBatch* e);

// fen stream
FenBatchStream* create_fen_batch_stream(int                  concurrency,
                                        int                  num_files,
                                        const char* const*   filenames,
                                        int                  batch_size,
                                        bool                 cyclic,
                                        DataloaderSkipConfig config,
                                        DataloaderDDPConfig  ddp_config);

void      destroy_fen_batch_stream(FenBatchStream* stream);
FenBatch* fetch_next_fen_batch(FenBatchStream* stream);

// sparse stream
SparseBatchStream* create_sparse_batch_stream(const char*          feature_set_c,
                                              int                  concurrency,
                                              int                  num_files,
                                              const char* const*   filenames,
                                              int                  batch_size,
                                              bool                 cyclic,
                                              DataloaderSkipConfig config,
                                              DataloaderDDPConfig  ddp_config);

void         destroy_sparse_batch_stream(SparseBatchStream* stream);
SparseBatch* fetch_next_sparse_batch(SparseBatchStream* stream);

}  // extern "C"