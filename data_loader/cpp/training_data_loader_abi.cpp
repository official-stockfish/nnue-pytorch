#include "training_data_loader_internal.h"
#include "training_data_loader_abi.h"

#include "lib/unique_counter.h"

using namespace binpack;
using namespace chess;

// TODO: We might want to introduce some exception safety to the abi.
// Although for our uses it doesn't have priority.
// Additionally the library could be quite unsafe since it reinterpret casts opaque pointers.
// The safest would be to track all "outgoing" pointers.

NNUE_API SparseBatch* NNUE_CDECL get_sparse_batch_from_fens(const char* feature_set_c,
                                               int                num_fens,
                                               const char* const* fens,
                                               int* scores,
                                               int* plies,
                                               int* results) {
    std::vector<TrainingDataEntry> entries;
    entries.reserve(num_fens);
    for (int i = 0; i < num_fens; ++i) {
        auto& e = entries.emplace_back();
        e.pos   = Position::fromFen(fens[i]);
        movegen::forEachLegalMove(e.pos, [&](Move m) { e.move = m; });
        e.score  = scores[i];
        e.ply    = plies[i];
        e.result = results[i];
    }

    auto feature = get_feature(feature_set_c);
    if (!feature)
        return nullptr;
    return new SparseBatch(*feature, entries);
}

NNUE_API FenBatchStream* NNUE_CDECL create_fen_batch_stream(int                  concurrency,
                                                     int                  num_files,
                                                     const char* const* filenames,
                                                     int                  batch_size,
                                                     bool                 cyclic,
                                                     DataloaderSkipConfig config,
                                                     DataloaderDDPConfig  ddp_config,
                                                     DataloaderHllConfig  hll_config) {
    (void)hll_config;
    auto skipPredicate = make_skip_predicate(config);
    auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

    return new FenBatchStream(concurrency, filenames_vec, batch_size, cyclic, skipPredicate,
                              ddp_config.rank, ddp_config.world_size, nullptr);
}

NNUE_API NNUE_COLD void NNUE_CDECL destroy_fen_batch_stream(FenBatchStream* stream) {
    delete stream;
}

NNUE_API SparseBatchStream* NNUE_CDECL create_sparse_batch_stream(const char* feature_set_c,
                                                             int                  concurrency,
                                                             int                  num_files,
                                                             const char* const* filenames,
                                                             int                  batch_size,
                                                             bool                 cyclic,
                                                             DataloaderSkipConfig config,
                                                             DataloaderDDPConfig  ddp_config,
                                                             DataloaderHllConfig  hll_config) {
    auto skipPredicate = make_skip_predicate(config);
    auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

    auto feature = get_feature(feature_set_c);
    if (!feature)
        return nullptr;

    auto counter = std::make_unique<nnue::UniquePositionCounter>(
        hll_config.initial_hll, hll_config.initial_hll_size, hll_config.initial_total, hll_config.initial_preskip);

    auto stream = new FeaturedBatchStream(std::move(feature), concurrency, filenames_vec, batch_size,
                                   cyclic, skipPredicate, ddp_config.rank, ddp_config.world_size, counter.release());
    return reinterpret_cast<SparseBatchStream*>(stream);
}

NNUE_API NNUE_COLD void NNUE_CDECL destroy_sparse_batch_stream(SparseBatchStream* stream) {
    delete reinterpret_cast<FeaturedBatchStream*>(stream);
}

NNUE_API SparseBatch* NNUE_CDECL fetch_next_sparse_batch(SparseBatchStream* stream) {
    return reinterpret_cast<FeaturedBatchStream*>(stream)->next();
}

NNUE_API FenBatch* NNUE_CDECL fetch_next_fen_batch(FenBatchStream* stream) {
    return stream->next();
}

NNUE_API void NNUE_CDECL destroy_sparse_batch(SparseBatch* e) {
    delete e;
}

NNUE_API void NNUE_CDECL destroy_fen_batch(FenBatch* e) {
    delete e;
}

// unique position counting (HLL)
NNUE_API void NNUE_CDECL get_unique_position_stats(SparseBatchStream* stream,
                                                   std::uint64_t*     out_preskip,
                                                   std::uint64_t*     out_total,
                                                   std::uint64_t*     out_unique) {
    auto* s = reinterpret_cast<FeaturedBatchStream*>(stream);
    const auto* counter = s->unique_counter();
    if (counter && out_preskip && out_total && out_unique)
        counter->stats(*out_preskip, *out_total, *out_unique);
    else { if (out_preskip) *out_preskip = 0; if (out_total) *out_total = 0; if (out_unique) *out_unique = 0; }
}

NNUE_API std::size_t NNUE_CDECL get_hll_state(SparseBatchStream* stream,
                                              std::uint8_t*      out_buf,
                                              std::size_t        buf_size) {
    auto* s = reinterpret_cast<FeaturedBatchStream*>(stream);
    const auto* counter = s->unique_counter();
    if (!counter) return 0;
    auto serialized = counter->serialize();
    if (serialized.size() > buf_size) return 0;
    std::memcpy(out_buf, serialized.data(), serialized.size());
    return serialized.size();
}

NNUE_API std::size_t NNUE_CDECL get_hll_state_size(SparseBatchStream* stream) {
    auto* s = reinterpret_cast<FeaturedBatchStream*>(stream);
    const auto* counter = s->unique_counter();
    if (!counter) return 0;
    return nnue::hll::HyperLogLog::kSerializedSize;
}

NNUE_API void NNUE_CDECL hll_count_from_state(const std::uint8_t* data,
                                               std::size_t         size,
                                               std::uint64_t*      out_count) {
    if (!data || !out_count || size < nnue::hll::HyperLogLog::kSerializedSize) {
        if (out_count) *out_count = 0;
        return;
    }
    try {
        auto h = nnue::hll::HyperLogLog::deserialize(data, size);
        *out_count = h.count();
    } catch (...) {
        *out_count = 0;
    }
}
