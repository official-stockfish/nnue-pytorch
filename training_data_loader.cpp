#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#if defined (__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif
#endif

using namespace binpack;
using namespace chess;

static Square orient(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return sq.flippedVertically().flippedHorizontally();
    }
}

static Square orient_flip(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        return sq.flippedVertically();
    }
}

struct HalfKP {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
        auto ksq = pos.kingSquare(color);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_SQ * HalfKP::NUM_PT;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto [start_j, offset] = HalfKP::fill_features_sparse(e, features, values, color);
        int j = start_j;
        auto& pos = e.pos;
        {
            // king square factor
            auto ksq = pos.kingSquare(color);
            features[j] = offset + static_cast<int>(orient(color, ksq));
            values[j] = static_cast<float>(start_j);
            ++j;
        }
        offset += K_INPUTS;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKP::NUM_SQ) + static_cast<int>(orient(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKA {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 12;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
    static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKA::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2 {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 11;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        if (p_idx == 11)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2Factorized {
    // Factorized features
    static constexpr int NUM_PT = 12;
    static constexpr int PIECE_INPUTS = HalfKAv2::NUM_SQ * NUM_PT;
    static constexpr int INPUTS = HalfKAv2::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

// ksq must not be oriented
static Square orient_flip_2(Color color, Square sq, Square ksq)
{
    bool h = ksq.file() < fileE;
    if (color == Color::Black)
        sq = sq.flippedVertically();
    if (h)
        sq = sq.flippedHorizontally();
    return sq;
}

struct HalfKAv2_hm {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 11;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ / 2;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static constexpr int KingBuckets[64] = {
      -1, -1, -1, -1, 31, 30, 29, 28,
      -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20,
      -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12,
      -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4,
      -1, -1, -1, -1, 3, 2, 1, 0
    };

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        Square o_ksq = orient_flip_2(color, ksq, ksq);
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        if (p_idx == 11)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ + KingBuckets[static_cast<int>(o_ksq)] * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, ksq, sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2_hmFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKAv2_hm::NUM_SQ * HalfKAv2_hm::NUM_PT;
    static constexpr int INPUTS = HalfKAv2_hm::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2_hm::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2_hm::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2_hm::NUM_SQ) + static_cast<int>(orient_flip_2(color, sq, ksq));
            ++j;
        }

        return { j, INPUTS };
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        return T::fill_features_sparse(e, features, values, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        psqt_indices = new int[size];
        layer_stack_indices = new int[size];

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* psqt_indices;
    int* layer_stack_indices;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] psqt_indices;
        delete[] layer_stack_indices;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos.piecesBB().count() - 1) / 4;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::White)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::Black)
            .first;
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    Stream(
        int concurrency,
        const char* filename,
        bool cyclic,
        std::function<bool(const TrainingDataEntry&)> skipPredicate,
        std::function<void(std::vector<TrainingDataEntry>&)> bufferFilter
    ) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename, cyclic, std::move(skipPredicate), std::move(bufferFilter)))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(
        int concurrency,
        const char* filename,
        bool cyclic,
        std::function<bool(const TrainingDataEntry&)> skipPredicate,
        std::function<void(std::vector<TrainingDataEntry>&)> bufferFilter
    ) :
        BaseType(1, filename, cyclic, std::move(skipPredicate), std::move(bufferFilter))
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(
        int concurrency,
        const char* filename,
        int batch_size,
        bool cyclic,
        std::function<bool(const TrainingDataEntry&)> skipPredicate,
        std::function<void(std::vector<TrainingDataEntry>&)> bufferFilter
    ) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
            cyclic,
            std::move(skipPredicate),
            std::move(bufferFilter)
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new StorageT(FeatureSet{}, entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

// Very simple fixed size string wrapper with a stable ABI to pass to python.
struct Fen
{
    Fen() :
        m_fen(nullptr)
    {
    }

    Fen(const std::string& fen) :
        m_size(fen.size()),
        m_fen(new char[fen.size() + 1])
    {
        std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    }

    Fen& operator=(const std::string& fen)
    {
        if (m_fen != nullptr)
        {
            delete m_fen;
        }

        m_size = fen.size();
        m_fen = new char[fen.size() + 1];
        std::memcpy(m_fen, fen.c_str(), fen.size() + 1);

        return *this;
    }

    ~Fen()
    {
        delete[] m_fen;
    }

private:
    int m_size;
    char* m_fen;
};

struct FenBatch
{
    FenBatch(const std::vector<TrainingDataEntry>& entries) :
        m_size(entries.size()),
        m_fens(new Fen[entries.size()])
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_fens[i] = entries[i].pos.fen();
        }
    }

    ~FenBatch()
    {
        delete[] m_fens;
    }

private:
    int m_size;
    Fen* m_fens;
};

struct FenBatchStream : Stream<FenBatch>
{
    static constexpr int num_feature_threads_per_reading_thread = 2;

    using BaseType = Stream<FenBatch>;

    FenBatchStream(int concurrency, const char* filename, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
            cyclic,
            skipPredicate,
            nullptr // TODO: if needed
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new FenBatch(entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    FenBatch* next()
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FenBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<FenBatch*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

std::function<void(std::vector<TrainingDataEntry>&)> make_buffer_filter()
{
    return [
        ](std::vector<TrainingDataEntry>& buffer) {
        static constexpr double desired_piece_count_weights[33] = {
            1.000000,
            1.121094, 1.234375, 1.339844, 1.437500, 1.527344, 1.609375, 1.683594, 1.750000,
            1.808594, 1.859375, 1.902344, 1.937500, 1.964844, 1.984375, 1.996094, 2.000000,
            1.996094, 1.984375, 1.964844, 1.937500, 1.902344, 1.859375, 1.808594, 1.750000,
            1.683594, 1.609375, 1.527344, 1.437500, 1.339844, 1.234375, 1.121094, 1.000000
        };

        static constexpr double desired_piece_count_weights_total = [](){
            double tot = 0;
            for (auto w : desired_piece_count_weights)
                tot += w;
            return tot;
        }();

        // Basically ignore n lower piece counts.
        static constexpr int nth_base = 3;
        static constexpr int max_skipping_factor = 5;

        int pc_hist[33] = {0};

        for (const auto& entry : buffer)
        {
            const int pc = entry.pos.piecesBB().count();
            pc_hist[pc] += 1;
        }

        std::vector<std::pair<int, double>> adjusted_hist_by_pc;
        for (int i = 2; i <= 32; ++i)
        {
            adjusted_hist_by_pc.emplace_back(i, pc_hist[i] / desired_piece_count_weights[i]);
        }
        std::sort(adjusted_hist_by_pc.begin(), adjusted_hist_by_pc.end(), [](const std::pair<int, double>& lhs, const std::pair<int, double>& rhs){
            return lhs.second < rhs.second;
        });
        const int base = adjusted_hist_by_pc[nth_base].second;
        int pc_hist_desired[33] = {0};
        for (int i = 2; i <= 32; ++i)
        {
            pc_hist_desired[i] = std::max(
                static_cast<int>(base * desired_piece_count_weights[i]),
                static_cast<int>(pc_hist[i] / max_skipping_factor)
            );
        }

        auto begin = buffer.begin();
        auto end = buffer.end();
        while (begin != end)
        {
            const int pc = begin->pos.piecesBB().count();
            if (pc_hist_desired[pc] > 0)
            {
                pc_hist_desired[pc] -= 1;
                ++begin;
            }
            else
            {
                --end;
                std::swap(*begin, *end);
            }
        }
        begin = buffer.begin(); // reassign begin because we were using it as a current pointer

        constexpr bool do_debug_print = false;
        if (do_debug_print) {
            int pc_hist2[33] = {0};
            for (auto b = begin; b != end; ++b)
            {
                const int pc = b->pos.piecesBB().count();
                pc_hist2[pc] += 1;
            }
            std::cout << "Total : " << buffer.size() << '\n';
            std::cout << "Passed: " << (end - begin) << '\n';
            for (int i = 0; i < 33; ++i)
                std::cout << i << ' ' << pc_hist2[i] << '\n';
        }
        buffer.resize(end - begin);
    };
}

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(bool filtered, int random_fen_skipping, bool wld_filtered)
{
    if (filtered || random_fen_skipping || wld_filtered)
    {
        return [
            random_fen_skipping,
            prob = double(random_fen_skipping) / (random_fen_skipping + 1),
            filtered,
            wld_filtered
            ](const TrainingDataEntry& e){

            static thread_local std::mt19937 gen(std::random_device{}());

            auto do_wld_skip = [&]() {
                std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return (e.isCapturingMove() || e.isInCheck());
            };

            if (random_fen_skipping && do_skip())
                return true;

            if (filtered && do_filter())
                return true;

            if (wld_filtered && do_wld_skip())
                return true;

            return false;
        };
    }

    return nullptr;
}

extern "C" {

    EXPORT SparseBatch* get_sparse_batch_from_fens(
        const char* feature_set_c,
        int num_fens,
        const char* const* fens,
        int* scores,
        int* plies,
        int* results
    )
    {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(num_fens);
        for (int i = 0; i < num_fens; ++i)
        {
            auto& e = entries.emplace_back();
            e.pos = Position::fromFen(fens[i]);
            movegen::forEachLegalMove(e.pos, [&](Move m){e.move = m;});
            e.score = scores[i];
            e.ply = plies[i];
            e.result = results[i];
        }

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new SparseBatch(FeatureSet<HalfKP>{}, entries);
        }
        else if (feature_set == "HalfKP^")
        {
            return new SparseBatch(FeatureSet<HalfKPFactorized>{}, entries);
        }
        else if (feature_set == "HalfKA")
        {
            return new SparseBatch(FeatureSet<HalfKA>{}, entries);
        }
        else if (feature_set == "HalfKA^")
        {
            return new SparseBatch(FeatureSet<HalfKAFactorized>{}, entries);
        }
        else if (feature_set == "HalfKAv2")
        {
            return new SparseBatch(FeatureSet<HalfKAv2>{}, entries);
        }
        else if (feature_set == "HalfKAv2^")
        {
            return new SparseBatch(FeatureSet<HalfKAv2Factorized>{}, entries);
        }
        else if (feature_set == "HalfKAv2_hm")
        {
            return new SparseBatch(FeatureSet<HalfKAv2_hm>{}, entries);
        }
        else if (feature_set == "HalfKAv2_hm^")
        {
            return new SparseBatch(FeatureSet<HalfKAv2_hmFactorized>{}, entries);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    // changing the signature needs matching changes in nnue_dataset.py
    EXPORT FenBatchStream* CDECL create_fen_batch_stream(int concurrency, const char* filename, int batch_size, bool cyclic, bool filtered, int random_fen_skipping, bool wld_filtered)
    {
        auto skipPredicate = make_skip_predicate(filtered, random_fen_skipping, wld_filtered);

        return new FenBatchStream(concurrency, filename, batch_size, cyclic, skipPredicate);
    }

    EXPORT void CDECL destroy_fen_batch_stream(FenBatchStream* stream)
    {
        delete stream;
    }

    // changing the signature needs matching changes in nnue_dataset.py
    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic,
                                                                 bool filtered, int random_fen_skipping, bool wld_filtered)
    {
        auto skipPredicate = make_skip_predicate(filtered, random_fen_skipping, wld_filtered);
        auto bufferFilter = make_buffer_filter();

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate, bufferFilter);
        }
        else if (feature_set == "HalfKP^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate, bufferFilter);
        }
        else if (feature_set == "HalfKA")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate, bufferFilter);
        }
        else if (feature_set == "HalfKA^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate, bufferFilter);
        }
        else if (feature_set == "HalfKAv2")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate, bufferFilter);
        }
        else if (feature_set == "HalfKAv2^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2Factorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate, bufferFilter);
        }
        else if (feature_set == "HalfKAv2_hm")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2_hm>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate, bufferFilter);
        }
        else if (feature_set == "HalfKAv2_hm^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2_hmFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate, bufferFilter);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT FenBatch* CDECL fetch_next_fen_batch(Stream<FenBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

    EXPORT void CDECL destroy_fen_batch(FenBatch* e)
    {
        delete e;
    }

}

/* benches */ //*
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP", 4, "10m_d3_q_2.binpack", 8192, true, false, 0, false);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
//*/
