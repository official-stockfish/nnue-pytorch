#include "training_data_loader_internal.h"
#include "training_data_loader_abi.h"

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
                                                     DataloaderDDPConfig  ddp_config) {
    auto skipPredicate = make_skip_predicate(config);
    auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

    return new FenBatchStream(concurrency, filenames_vec, batch_size, cyclic, skipPredicate,
                              ddp_config.rank, ddp_config.world_size);
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
                                                             DataloaderDDPConfig  ddp_config) {
    auto skipPredicate = make_skip_predicate(config);
    auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

    auto feature = get_feature(feature_set_c);
    if (!feature)
        return nullptr;
    auto stream = new FeaturedBatchStream(std::move(feature), concurrency, filenames_vec, batch_size,
                                   cyclic, skipPredicate, ddp_config.rank, ddp_config.world_size);
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
