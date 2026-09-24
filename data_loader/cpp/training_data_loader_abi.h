#pragma once

#include <cstddef>
#include <cstdint>

#include "nnue_macros.h"
#include "training_data_loader_structs.h"

// Opaque handles
struct SparseBatch;
struct FenBatch;
struct FenBatchStream;
struct SparseBatchStream;

// batches
NNUE_API SparseBatch* NNUE_CDECL get_sparse_batch_from_fens(const char*        feature_set_c,
                                        int                num_fens,
                                        const char* const* fens,
                                        int*               scores,
                                        int*               plies,
                                        int*               results);

NNUE_API void NNUE_CDECL destroy_sparse_batch(SparseBatch* e);
NNUE_API void NNUE_CDECL destroy_fen_batch(FenBatch* e);

// fen stream
NNUE_API FenBatchStream* NNUE_CDECL create_fen_batch_stream(int                  concurrency,
                                        int                  num_files,
                                        const char* const*   filenames,
                                        int                  batch_size,
                                        bool                 cyclic,
                                        DataloaderSkipConfig config,
                                        DataloaderDDPConfig  ddp_config,
                                        DataloaderHllConfig  hll_config);

NNUE_API void      NNUE_CDECL destroy_fen_batch_stream(FenBatchStream* stream);
NNUE_API FenBatch* NNUE_CDECL fetch_next_fen_batch(FenBatchStream* stream);

// sparse stream
NNUE_API SparseBatchStream* NNUE_CDECL create_sparse_batch_stream(const char*          feature_set_c,
                                              int                  concurrency,
                                              int                  num_files,
                                              const char* const*   filenames,
                                              int                  batch_size,
                                              bool                 cyclic,
                                              DataloaderSkipConfig config,
                                              DataloaderDDPConfig  ddp_config,
                                              DataloaderHllConfig  hll_config);

NNUE_API void         NNUE_CDECL destroy_sparse_batch_stream(SparseBatchStream* stream);
NNUE_API SparseBatch* NNUE_CDECL fetch_next_sparse_batch(SparseBatchStream* stream);

// unique position counting (HLL)
// Write the preskip count (before filtering), total count (after
// filtering), and approximate unique count into the output params.
// Race-free; may be called while the stream is producing batches.
NNUE_API void NNUE_CDECL get_unique_position_stats(SparseBatchStream* stream,
                                                    std::uint64_t*     out_preskip,
                                                    std::uint64_t*     out_total,
                                                    std::uint64_t*     out_unique);

// Serialize the current HLL state into out_buf (at most buf_size bytes).
// Returns the number of bytes written, or 0 if buf_size is too small.
// The caller should allocate at least get_hll_state_size() bytes.
NNUE_API std::size_t NNUE_CDECL get_hll_state(SparseBatchStream* stream,
                                               std::uint8_t*      out_buf,
                                               std::size_t        buf_size);

// Returns the size in bytes needed to serialize the HLL state
// (16-byte header + 2^20 registers ≈ 1 MiB).
NNUE_API std::size_t NNUE_CDECL get_hll_state_size(SparseBatchStream* stream);

// Deserialize an HLL state (as produced by get_hll_state) and compute
// the approximate unique count. Returns 0 if data is invalid.
NNUE_API void NNUE_CDECL hll_count_from_state(const std::uint8_t* data,
                                               std::size_t         size,
                                               std::uint64_t*      out_count);
