#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <string_view>
#include <utility>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "training_data_loader_structs.h"

struct IFeatureExtractor {
    virtual ~IFeatureExtractor() = default;
    virtual int inputs() const = 0;
    virtual int max_active_features() const = 0;
    virtual std::pair<int, int> fill_features_sparse(const struct binpack::TrainingDataEntry& e,
                                                     int* features,
                                                     float* values,
                                                     chess::Color color) const = 0;
};

std::shared_ptr<IFeatureExtractor> get_feature(std::string_view name);
std::function<bool(const struct binpack::TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config);

struct SparseBatch final {
    static constexpr bool IS_BATCH = true;

    SparseBatch(const IFeatureExtractor& feature_set, const std::vector<struct binpack::TrainingDataEntry>& entries);
    ~SparseBatch();

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int    num_active_white_features;
    int    num_active_black_features;
    int    max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* psqt_indices;
    int* layer_stack_indices;

private:
    void fill_entry(const IFeatureExtractor& fs, int i, const struct binpack::TrainingDataEntry& e);
    void fill_features(const IFeatureExtractor& fs, int i, const struct binpack::TrainingDataEntry& e);
};

struct AnyStream {
    virtual ~AnyStream() = default;
};

template<typename StorageT>
struct Stream: AnyStream {
    using StorageType = StorageT;

    Stream(int concurrency,
           const std::vector<std::string>& filenames,
           bool cyclic,
           std::function<bool(const struct binpack::TrainingDataEntry&)> skipPredicate,
           int rank = 0,
           int world_size = 1) :
        m_stream(training_data::open_sfen_input_file_parallel(
          concurrency, filenames, cyclic, skipPredicate, rank, world_size)) {}

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

struct FeaturedBatchStream: Stream<SparseBatch> {
    using BaseType = Stream<SparseBatch>;
    static constexpr double worker_thread_ratio = 0.20;

    FeaturedBatchStream(std::shared_ptr<IFeatureExtractor> feature_set,
                        int concurrency,
                        const std::vector<std::string>& filenames,
                        int batch_size,
                        bool cyclic,
                        std::function<bool(const struct binpack::TrainingDataEntry&)> skipPredicate,
                        int rank = 0,
                        int world_size = 1);
    ~FeaturedBatchStream() final;

    SparseBatch* next() override;

private:
    std::shared_ptr<IFeatureExtractor> m_feature_set;
    int m_batch_size;
    int m_concurrency;
    std::deque<SparseBatch*> m_batches;
    std::mutex m_batch_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;
    std::vector<std::thread> m_workers;

    static int calculate_num_reader_threads(int concurrency);
    static int calculate_num_worker_threads(int concurrency);
};

struct Fen final {
    Fen();
    Fen(const std::string& fen);
    Fen& operator=(const std::string& fen);
    ~Fen();

private:
    int m_size;
    char* m_fen;
};

struct FenBatch final {
    FenBatch(const std::vector<struct binpack::TrainingDataEntry>& entries);
    ~FenBatch();

private:
    int m_size;
    Fen* m_fens;
};

struct FenBatchStream: Stream<FenBatch> {
    using BaseType = Stream<FenBatch>;
    static constexpr double worker_thread_ratio = 0.5;

    FenBatchStream(int concurrency,
                   const std::vector<std::string>& filenames,
                   int batch_size,
                   bool cyclic,
                   std::function<bool(const struct binpack::TrainingDataEntry&)> skipPredicate,
                   int rank = 0,
                   int world_size = 1);
    ~FenBatchStream() final;

    FenBatch* next() override;

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<FenBatch*> m_batches;
    std::mutex m_batch_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;
    std::vector<std::thread> m_workers;

    static int calculate_num_reader_threads(int concurrency);
    static int calculate_num_worker_threads(int concurrency);
};
