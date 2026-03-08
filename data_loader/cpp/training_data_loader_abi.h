#pragma once

#include <cstddef>

#include "nnue_macros.h"
#include "training_data_loader_structs.h"

// Opaque handles
struct SparseBatch;
struct FenBatch;
struct FenBatchStream;
struct SparseBatchStream;

// batches
NNUE_ABI SparseBatch* CDECL get_sparse_batch_from_fens(const char*        feature_set_c,
                                        int                num_fens,
                                        const char* const* fens,
                                        int*               scores,
                                        int*               plies,
                                        int*               results);

NNUE_ABI void CDECL destroy_sparse_batch(SparseBatch* e);
NNUE_ABI void CDECL destroy_fen_batch(FenBatch* e);

// fen stream
NNUE_ABI FenBatchStream* CDECL create_fen_batch_stream(int                  concurrency,
                                        int                  num_files,
                                        const char* const*   filenames,
                                        int                  batch_size,
                                        bool                 cyclic,
                                        DataloaderSkipConfig config,
                                        DataloaderDDPConfig  ddp_config);

NNUE_ABI void      CDECL destroy_fen_batch_stream(FenBatchStream* stream);
NNUE_ABI FenBatch* CDECL fetch_next_fen_batch(FenBatchStream* stream);

// sparse stream
NNUE_ABI SparseBatchStream* CDECL create_sparse_batch_stream(const char*          feature_set_c,
                                              int                  concurrency,
                                              int                  num_files,
                                              const char* const*   filenames,
                                              int                  batch_size,
                                              bool                 cyclic,
                                              DataloaderSkipConfig config,
                                              DataloaderDDPConfig  ddp_config);

NNUE_ABI void         CDECL destroy_sparse_batch_stream(SparseBatchStream* stream);
NNUE_ABI SparseBatch* CDECL fetch_next_sparse_batch(SparseBatchStream* stream);
