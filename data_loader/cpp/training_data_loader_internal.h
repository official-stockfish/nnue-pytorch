#pragma once

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <deque>
#include <mutex>
#include <optional>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <string_view>
#include <utility>

#include "lib/parallel_dataloader.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/unique_counter.h"
#include "training_data_loader_structs.h"

struct IFeatureExtractor {
    virtual ~IFeatureExtractor()                                               = default;
    virtual int                 inputs() const                                 = 0;
    virtual int                 max_active_features() const                    = 0;
    virtual std::pair<int, int> fill_features_sparse(const struct binpack::TrainingDataEntry& e,
                                                     int*         features,
                                                     chess::Color color) const = 0;
};

std::shared_ptr<IFeatureExtractor> get_feature(std::string_view name);

// Layer-stack index of a position from the keys named in --ls-keys (mixed radix, the last key
// varying fastest), as in Network::layer_stack().
struct LayerStackKeys {
    struct Key {
        int states;
        int (*value)(const chess::Position& pos);
    };
    std::vector<Key> keys;

    int index(const chess::Position& pos) const;
};

std::optional<LayerStackKeys> get_layer_stack_keys(std::string_view names);

std::function<bool(const struct binpack::TrainingDataEntry&)>
make_skip_predicate(DataloaderSkipConfig config);

// Keep large feature buffers resident between batches. Returning ~300 MiB to
// the system allocator on every batch can stall the Python prefetch thread.
// Each stream owns a bounded cache; outstanding batches share its lifetime.
class SparseBatchBufferPool final {
   public:
    using Buffers = std::pair<std::unique_ptr<float[]>, std::unique_ptr<int[]>>;

    // Packed index buffers vary in size from batch to batch: a buffer is reused
    // by any batch that fits in its capacity.
    struct Packed {
        std::unique_ptr<int[]> data;
        std::size_t            capacity = 0;
    };

    SparseBatchBufferPool(std::size_t floats, std::size_t ints, std::size_t capacity);
    Buffers acquire(std::size_t floats, std::size_t ints);
    void    release(Buffers buffers, std::size_t floats, std::size_t ints);
    Packed  acquire_packed(std::size_t ints);
    void    release_packed(Packed packed);

   private:
    const std::size_t    m_floats;
    const std::size_t    m_ints;
    const std::size_t    m_capacity;
    std::mutex           m_mutex;
    std::vector<Buffers> m_free;
    std::vector<Packed>  m_free_packed;
};

struct SparseBatch final {
    static constexpr bool IS_BATCH = true;

    SparseBatch(const IFeatureExtractor&                              feature_set,
                const LayerStackKeys&                                 ls_keys,
                const std::vector<struct binpack::TrainingDataEntry>& entries,
                std::shared_ptr<SparseBatchBufferPool>                buffer_pool = nullptr,
                bool                                                  prepack     = false);
    ~SparseBatch();

    // Packed int32 layout: white offsets [size + 1], black offsets [size + 1],
    // white IDs [num_active_white_features], black IDs [num_active_black_features],
    // layer-stack indices [size]. Offsets begin at zero and end
    // at the total ID count.
    // capacity is in elements; insufficient capacity leaves output untouched.
    // A prepacked batch has white == black == nullptr; the worker built the
    // layout and this only copies it.
    bool pack_indices(int* output, std::size_t capacity) const;
    std::size_t packed_size() const;

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int    num_active_white_features;
    int    num_active_black_features;
    int    max_active_features;
    int*   white;
    int*   black;
    int*   layer_stack_index;

#ifdef NNUE_LOADER_STATISTICS
    std::vector<struct binpack::TrainingDataEntry> entries_copy;
#endif

   private:
    float*                                 m_float_block = nullptr;
    int*                                   m_int_block   = nullptr;
    std::shared_ptr<SparseBatchBufferPool> m_buffer_pool;
    SparseBatchBufferPool::Packed          m_packed;
    std::vector<int>                       m_feature_counts;
    void fill_targets(int i, const struct binpack::TrainingDataEntry& e);
    void fill_entry(const IFeatureExtractor& fs, int i, const struct binpack::TrainingDataEntry& e);
    void fill_packed(const IFeatureExtractor&                              fs,
                     const LayerStackKeys&                                 ls_keys,
                     const std::vector<struct binpack::TrainingDataEntry>& entries);
    void
    fill_features(const IFeatureExtractor& fs, int i, const struct binpack::TrainingDataEntry& e);
};

struct AnyStream {
    virtual ~AnyStream() = default;

    nnue::UniquePositionCounter*       unique_counter() { return m_unique_counter.get(); }
    const nnue::UniquePositionCounter* unique_counter() const { return m_unique_counter.get(); }

   protected:
    // Owned here (not in FeaturedBatchStream/FenBatchStream) so that it is
    // destroyed AFTER m_stream (C++ base-class destruction order): the
    // CompressedTrainingDataEntryParallelReader workers, which hold a raw
    // pointer to the counter, are joined when m_stream is destroyed.
    std::unique_ptr<nnue::UniquePositionCounter> m_unique_counter;
};

template<typename StorageT>
struct Stream: AnyStream {
    using StorageType = StorageT;

    Stream(int                                                           concurrency,
           const std::vector<std::string>&                               filenames,
           bool                                                          cyclic,
           std::function<bool(const struct binpack::TrainingDataEntry&)> skipPredicate,
           nnue::UniquePositionCounter*                                  counter    = nullptr,
           int                                                           rank       = 0,
           int                                                           world_size = 1,
           DataloaderIOConfig                                            io_config  = {},
           int                                                           num_consumer_threads = 4) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency,
                                                              filenames,
                                                              cyclic,
                                                              skipPredicate,
                                                              counter,
                                                              rank,
                                                              world_size,
                                                              io_config,
                                                              num_consumer_threads)) {
        if (counter)
            m_unique_counter.reset(counter);
    }

    virtual StorageT* next() = 0;

    std::size_t get_io_stats(DataloaderFileStats* out, std::size_t max_files) {
        return m_stream->get_io_stats(out, max_files);
    }

   protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

// Feature-extraction builders and the finished-batch deque live here;
// see lib/parallel_dataloader.h (top) for the full pipeline map.
struct FeaturedBatchStream final: Stream<SparseBatch> {
    using BaseType = Stream<SparseBatch>;
    // Builders set the pipeline's throughput. With the default feature set
    // (Full_Threats+PP_3Wide+K32H3^, B = 65536, 16 workers) the loader
    // delivered 36 batches/s with 1/4 of the workers as builders, 46 with
    // 3/8, 49 with 1/2 and 48 with 5/8. Larger batches need proportionally
    // more decode workers to keep the pipeline fed (B = 131072, 32 workers:
    // 42 batches/s with 1/4 of the workers as builders vs 28 with 1/2), so
    // the default ratio scales down with the batch size: 1/2 at B <= 65536
    // down to 1/4 at B >= 131072.
    static double worker_thread_ratio(int batch_size) {
        const double t = std::min(std::max(std::log2(double(batch_size)) - 16.0, 0.0), 1.0);
        return 0.5 - 0.25 * t;
    }

    FeaturedBatchStream(std::shared_ptr<IFeatureExtractor>                            feature_set,
                        LayerStackKeys                                                ls_keys,
                        int                                                           concurrency,
                        const std::vector<std::string>&                               filenames,
                        int                                                           batch_size,
                        bool                                                          cyclic,
                        std::function<bool(const struct binpack::TrainingDataEntry&)> skipPredicate,
                        int                                                           rank = 0,
                        int                          world_size                            = 1,
                        nnue::UniquePositionCounter* counter         = nullptr,
                        DataloaderIOConfig           io_config       = {},
                        bool                         prepack_indices = false);
    ~FeaturedBatchStream() final;

    SparseBatch* next() override;

   private:
    std::shared_ptr<IFeatureExtractor> m_feature_set;
    LayerStackKeys                     m_ls_keys;
    int                                m_batch_size;
    int                                m_concurrency;
    bool                               m_prepack_indices;
    std::deque<SparseBatch*>           m_batches;
    std::mutex                         m_batch_mutex;
    std::condition_variable            m_batches_not_full;
    std::condition_variable            m_batches_any;
    std::atomic_bool                   m_stop_flag;
    std::atomic_int                    m_num_workers;
    // Finished batches queued for the consumer. Decoupled from the worker
    // count: every queued batch costs batch_size*max_active_features*4
    // bytes (~300 MiB at production settings), so a large worker count must
    // not translate into a large queue.
    int                                    m_batch_queue_capacity;
    std::shared_ptr<SparseBatchBufferPool> m_buffer_pool;
    std::vector<std::thread>               m_workers;

    static int calculate_num_reader_threads(int concurrency, int batch_size);
    static int calculate_num_worker_threads(int concurrency, int batch_size);
};

struct Fen final {
    Fen();
    Fen(const std::string& fen);
    Fen& operator=(const std::string& fen);
    ~Fen();

   private:
    int   m_size;
    char* m_fen;
};

struct FenBatch final {
    FenBatch(const std::vector<struct binpack::TrainingDataEntry>& entries);
    ~FenBatch();

   private:
    int  m_size;
    Fen* m_fens;
};

struct FenBatchStream final: Stream<FenBatch> {
    using BaseType                              = Stream<FenBatch>;
    static constexpr double worker_thread_ratio = 0.5;

    FenBatchStream(int                                                           concurrency,
                   const std::vector<std::string>&                               filenames,
                   int                                                           batch_size,
                   bool                                                          cyclic,
                   std::function<bool(const struct binpack::TrainingDataEntry&)> skipPredicate,
                   int                                                           rank       = 0,
                   int                                                           world_size = 1,
                   nnue::UniquePositionCounter*                                  counter = nullptr,
                   DataloaderIOConfig                                            io_config = {});
    ~FenBatchStream() final;

    FenBatch* next() override;

   private:
    int                      m_batch_size;
    int                      m_concurrency;
    std::deque<FenBatch*>    m_batches;
    std::mutex               m_batch_mutex;
    std::condition_variable  m_batches_not_full;
    std::condition_variable  m_batches_any;
    std::atomic_bool         m_stop_flag;
    std::atomic_int          m_num_workers;
    int                      m_batch_queue_capacity;
    std::vector<std::thread> m_workers;

    static int calculate_num_reader_threads(int concurrency);
    static int calculate_num_worker_threads(int concurrency);
};
