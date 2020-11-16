#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"

#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif

using namespace binpack;
using namespace chess;

struct HalfKP {

    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;
    static constexpr int MAX_ACTIVE_FEATURES = 32;

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

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, int& counter, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
        auto ksq = pos.kingSquare(color);
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            int idx = counter * 2;
            counter += 1;
            features[idx] = i;
            features[idx + 1] = feature_index(color, orient(color, ksq), sq, p);
        }
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static void fill_features_sparse(int i, const TrainingDataEntry& e, int* features, int& counter, Color color)
    {
        T::fill_features_sparse(i, e, features, counter, color);
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
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2];

        num_active_white_features = 0;
        num_active_black_features = 0;

        std::memset(white, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));
        std::memset(black, 0, size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES * 2 * sizeof(int));

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
    int* white;
    int* black;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
    }

private:

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        FeatureSet<Ts...>::fill_features_sparse(i, e, white, num_active_white_features, Color::White);
        FeatureSet<Ts...>::fill_features_sparse(i, e, black, num_active_black_features, Color::Black);
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

    Stream(int concurrency, const char* filename, bool cyclic) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename, cyclic))
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

    AsyncStream(int concurrency, const char* filename, bool cyclic) :
        BaseType(1, filename, cyclic)
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

    FeaturedBatchStream(int concurrency, const char* filename, int batch_size, bool cyclic) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
            cyclic
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

template <template <typename, typename> typename StreamT, typename StorageT, typename... ArgsTs>
StreamT<FeatureSet<HalfKP>, StorageT>* create_stream(std::string feature_set, ArgsTs&&... args)
{
    if (feature_set == "HalfKP")
    {
        return new StreamT<FeatureSet<HalfKP>, StorageT>(std::forward<ArgsTs>(args)...);
    }
    else
    {
        return nullptr;
    }
}

extern "C" {

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set, int concurrency, const char* filename, int batch_size, bool cyclic)
    {
        return create_stream<FeaturedBatchStream, SparseBatch>(feature_set, concurrency, filename, batch_size, cyclic);
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

}

/* benches */ //*
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP", 4, "10m_d3_q_2.binpack", 8192, true);
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
