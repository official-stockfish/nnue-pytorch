#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <string_view>
#include <thread>
#include <deque>
#include <random>
#include <variant>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#if defined(__x86_64__)
    #define EXPORT
    #define CDECL
#else
    #if defined(_MSC_VER)
        #define EXPORT __declspec(dllexport)
        #define CDECL __cdecl
    #else
        #define EXPORT
        #define CDECL __attribute__((__cdecl__))
    #endif
#endif

using namespace binpack;
using namespace chess;

// ksq must not be oriented
static Square orient_flip_2(Color color, Square sq, Square ksq) {
    bool h = ksq.file() < fileE;
    if (color == Color::Black)
        sq = sq.flippedVertically();
    if (h)
        sq = sq.flippedHorizontally();
    return sq;
}

struct HalfKAv2_hm {
    static constexpr std::string_view NAME = "HalfKAv2_hm";

    static constexpr int NUM_SQ     = 64;
    static constexpr int NUM_PT     = 12;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS     = NUM_PLANES * NUM_SQ / 2;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    // clang-format off
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

    // clang-format on


    static int feature_index(Color color, Square ksq, Square sq, Piece p) {
        Square o_ksq = orient_flip_2(color, ksq, ksq);
        auto   p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ
             + KingBuckets[static_cast<int>(o_ksq)] * NUM_PLANES;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color) {
        auto& pos    = e.pos;
        auto  pieces = pos.piecesBB();
        auto  ksq    = pos.kingSquare(color);

        int j = 0;
        for (Square sq : pieces)
        {
            auto p      = pos.pieceAt(sq);
            values[j]   = 1.0f;
            features[j] = feature_index(color, ksq, sq, p);
            ++j;
        }

        return {j, INPUTS};
    }
};

constexpr int numvalidtargets[12] = {6, 6, 10, 10, 8, 8, 8, 8, 10, 10, 0, 0};

using ThreatOffsetTable = std::array<std::array<int, 66>, 12>;

struct ThreatFeatureCalculation {
    ThreatOffsetTable table;
    int totalfeatures;
};

constexpr auto threatfeaturecalc = []() {
    ThreatOffsetTable t{};

    constexpr auto pseudo_attacks = bb::detail::generatePseudoAttacks();
    int            pieceoffset    = 0;

    Piece piecetbl[12] = {whitePawn, blackPawn, whiteKnight, blackKnight, whiteBishop, blackBishop,
                          whiteRook, blackRook, whiteQueen,  blackQueen,  whiteKing,   blackKing};

    for (int c = 0; c < 2; c++)
    {
        for (int pt = 0; pt < 6; pt++)
        {
            int piece        = 2 * pt + c;
            t[piece][65]     = pieceoffset;
            int squareoffset = 0;
            for (int from = (int) a1; from <= (int) h8; from++)
            {
                t[piece][from] = squareoffset;
                if (piecetbl[piece].type() != PieceType::Pawn)
                {
                    Bitboard attacks = pseudo_attacks[piecetbl[piece].type()][Square(from)];
                    squareoffset += attacks.count();
                }
                else if (from >= (int) a2 && from <= (int) h7)
                {
                    Bitboard attacks =
                      bb::pawnAttacks(Bitboard::square(Square(from)), piecetbl[piece].color());
                    squareoffset += attacks.count();
                }
            }
            t[piece][64] = squareoffset;
            pieceoffset += numvalidtargets[piece] * squareoffset;
        }
    }

    return ThreatFeatureCalculation{t, pieceoffset};
}();

constexpr ThreatOffsetTable threatoffsets = threatfeaturecalc.table;
constexpr int threatfeatures = threatfeaturecalc.totalfeatures;
static_assert(threatfeatures == 60144);

struct Full_Threats {
    static constexpr std::string_view NAME = "Full_Threats";

    static constexpr int SQUARE_NB           = 64;
    static constexpr int PIECE_NB            = 12;
    static constexpr int COLOR_NB            = 2;
    static constexpr int PIECE_TYPE_NB       = 6;
    static constexpr int MAX_ACTIVE_FEATURES = 128 + 32;

    // clang-format off
    static constexpr Square OrientTBL[COLOR_NB][SQUARE_NB] = {
      { a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1,
        a1, a1, a1, a1, h1, h1, h1, h1 },
      { a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8,
        a8, a8, a8, a8, h8, h8, h8, h8 }
    };

    static constexpr int map[PIECE_TYPE_NB][PIECE_TYPE_NB] = {
      {0, 1, -1, 2, -1, -1},
      {0, 1, 2, 3, 4, -1},
      {0, 1, 2, 3, -1, -1},
      {0, 1, 2, 3, -1, -1},
      {0, 1, 2, 3, 4, -1},
      {-1, -1, -1, -1, -1, -1}
    };

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
    // clang-format on

    static constexpr int NUM_SQ     = 64;
    static constexpr int NUM_PT     = 12;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;

    static constexpr int NUM_THREAT_FEATURES = threatfeatures;
    static constexpr int INPUTS     = NUM_THREAT_FEATURES + NUM_PLANES * NUM_SQ / 2;


    static int psq_index(Color color, Square ksq, Square sq, Piece p) {
        Square o_ksq = orient_flip_2(color, ksq, ksq);
        auto   p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return NUM_THREAT_FEATURES + static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ
             + KingBuckets[static_cast<int>(o_ksq)] * NUM_PLANES;
    }

    static int
    threat_index(Color Perspective, Piece attkr, Square from, Square to, Piece attkd, Square ksq) {
        bool enemy = (attkr.color() != attkd.color());
        from       = (Square) (int(from) ^ (int) OrientTBL[(int) Perspective][(int) ksq]);
        to         = (Square) (int(to) ^ (int) OrientTBL[(int) Perspective][(int) ksq]);
        if (Perspective == Color::Black)
        {
            attkr = Piece::fromId((int) attkr ^ 1);
            attkd = Piece::fromId((int) attkd ^ 1);
        }
        if ((map[(int) attkr.type()][(int) attkd.type()] < 0)
            || (attkr.type() == attkd.type() && (enemy || attkr.type() != PieceType::Pawn)
                && from < to))
        {
            return -1;
        }
        Bitboard attacks = (attkr.type() == PieceType::Pawn)
                           ? bb::pawnAttacks(Bitboard::square(Square(from)), attkr.color())
                           : bb::detail::pseudoAttacks()[attkr.type()][Square(from)];
        Bitboard upto    = Bitboard::square(to);
        return int(threatoffsets[(int) attkr][65]
                   + (int(attkd.color()) * (numvalidtargets[(int) attkr] / 2)
                      + map[(int) attkr.type()][(int) attkd.type()])
                       * threatoffsets[(int) attkr][64]
                   + threatoffsets[(int) attkr][(int) from]
                   + (Bitboard::fromBits((1ULL << (int) to) - 1) & attacks).count());
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color) {
        auto& pos         = e.pos;
        auto  pieces      = pos.piecesBB();
        auto  ksq         = pos.kingSquare(color);
        Color order[2][2] = {{Color::White, Color::Black}, {Color::Black, Color::White}};
        int   k           = 0;
        for (int i = (int) Color::White; i <= (int) Color::Black; i++)
        {
            for (int j = (int) PieceType::Pawn; j <= (int) PieceType::King; j++)
            {
                Color     c     = order[(int) color][i];
                PieceType pt    = PieceType(j);
                Piece     attkr = Piece(pt, c);
                Bitboard  bb    = pos.piecesBB(attkr);
                if (pt == PieceType::Pawn)
                {
                    auto right         = (c == Color::White) ? Offset(1, 1) : Offset(-1, -1);
                    auto left          = (c == Color::White) ? Offset(-1, 1) : Offset(1, -1);
                    auto attacks_left  = bb.shifted(right) & pieces;
                    auto attacks_right = bb.shifted(left) & pieces;
                    for (Square to : attacks_left)
                    {
                        Square from  = Square((int) to - (c == Color::White ? 9 : -9));
                        Piece  attkd = pos.pieceAt(to);
                        int    index = threat_index(color, attkr, from, to, attkd, ksq);
                        if (index >= 0)
                        {
                            values[k]   = 1.0f;
                            features[k] = index;
                            k++;
                        }
                    }
                    for (Square to : attacks_right)
                    {
                        Square from  = Square((int) to - (c == Color::White ? 7 : -7));
                        Piece  attkd = pos.pieceAt(to);
                        int    index = threat_index(color, attkr, from, to, attkd, ksq);
                        if (index >= 0)
                        {
                            values[k]   = 1.0f;
                            features[k] = index;
                            k++;
                        }
                    }
                    for (Square from : bb)
                    {
                        values[k]   = 1.0f;
                        features[k] = psq_index(color, ksq, from, attkr);
                        k++;
                    }
                }
                else
                {
                    for (Square from : bb)
                    {
                        values[k]   = 1.0f;
                        features[k] = psq_index(color, ksq, from, attkr);
                        k++;
                        Bitboard attacks = pos.attacks(from) & pieces;
                        for (Square to : attacks)
                        {
                            Piece attkd = pos.pieceAt(to);
                            int   index = threat_index(color, attkr, from, to, attkd, ksq);
                            if (index >= 0)
                            {
                                values[k]   = 1.0f;
                                features[k] = index;
                                k++;
                            }
                        }
                    }
                }
            }
        }

        return {k, INPUTS};
    }
};

template<typename T, typename... Ts>
struct FeatureSet {
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS              = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static constexpr std::string_view NAME = T::NAME;

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color) {
        return T::fill_features_sparse(e, features, values, color);
    }
};


template<typename... Ts>
auto find_feature(std::string_view name) {
    using Variant = std::variant<std::monostate, Ts...>;
    using Factory = Variant (*)();

    struct Entry {
        std::string_view name;
        Factory          make;
    };

    static constexpr Entry factories[] = {{Ts::NAME, +[]() -> Variant { return Ts{}; }}...};

    for (auto& f : factories)
    {
        if (name == f.name)
            return f.make();
    }

    return Variant{std::monostate{}};
}

auto get_feature(std::string_view name) {
    return find_feature<HalfKAv2_hm,    //
                        Full_Threats     //
                        >(name);
}

struct SparseBatch {
    static constexpr bool IS_BATCH = true;

    template<typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries) {
        num_inputs          = FeatureSet<Ts...>::INPUTS;
        size                = entries.size();
        is_white            = new float[size];
        outcome             = new float[size];
        score               = new float[size];
        white               = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black               = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values        = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values        = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        psqt_indices        = new int[size];
        layer_stack_indices = new int[size];

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features       = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for (int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

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
    float* white_values;
    float* black_values;
    int*   psqt_indices;
    int*   layer_stack_indices;

    ~SparseBatch() {
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
    template<typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e) {
        is_white[i]            = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i]             = (e.result + 1.0f) / 2.0f;
        score[i]               = e.score;
        psqt_indices[i]        = (e.pos.piecesBB().count() - 1) / 4;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template<typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e) {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;
        num_active_white_features += FeatureSet<Ts...>::fill_features_sparse(
                                       e, white + offset, white_values + offset, Color::White)
                                       .first;
        num_active_black_features += FeatureSet<Ts...>::fill_features_sparse(
                                       e, black + offset, black_values + offset, Color::Black)
                                       .first;
    }
};

struct AnyStream {
    virtual ~AnyStream() = default;
};

template<typename StorageT>
struct Stream: AnyStream {
    using StorageType = StorageT;

    Stream(int                                           concurrency,
           const std::vector<std::string>&               filenames,
           bool                                          cyclic,
           std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(
          concurrency, filenames, cyclic, skipPredicate)) {}

    virtual StorageT* next() = 0;

   protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template<typename StorageT>
struct AsyncStream: Stream<StorageT> {
    using BaseType = Stream<StorageT>;

    AsyncStream(int                                           concurrency,
                const std::vector<std::string>&               filenames,
                bool                                          cyclic,
                std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filenames, cyclic, skipPredicate) {}

    ~AsyncStream() {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

   protected:
    std::future<StorageT*> m_next;
};

template<typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream: Stream<StorageT> {
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType   = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int                                           concurrency,
                        const std::vector<std::string>&               filenames,
                        int                                           batch_size,
                        bool                                          cyclic,
                        std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(std::max(1, concurrency / num_feature_threads_per_reading_thread),
                 filenames,
                 cyclic,
                 skipPredicate),
        m_concurrency(concurrency),
        m_batch_size(batch_size) {
        m_stop_flag.store(false);

        auto worker = [this]() {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while (!m_stop_flag.load())
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
                    m_batches_not_full.wait(lock, [this]() {
                        return m_batches.size() < m_concurrency + 1 || m_stop_flag.load();
                    });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }
            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
          1, concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread));

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

    StorageT* next() override {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock,
                           [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

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

    ~FeaturedBatchStream() {
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
    int                     m_batch_size;
    int                     m_concurrency;
    std::deque<StorageT*>   m_batches;
    std::mutex              m_batch_mutex;
    std::mutex              m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool        m_stop_flag;
    std::atomic_int         m_num_workers;

    std::vector<std::thread> m_workers;
};

// Very simple fixed size string wrapper with a stable ABI to pass to python.
struct Fen {
    Fen() :
        m_fen(nullptr) {}

    Fen(const std::string& fen) :
        m_size(fen.size()),
        m_fen(new char[fen.size() + 1]) {
        std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    }

    Fen& operator=(const std::string& fen) {
        if (m_fen != nullptr)
        {
            delete[] m_fen;
        }

        m_size = fen.size();
        m_fen  = new char[fen.size() + 1];
        std::memcpy(m_fen, fen.c_str(), fen.size() + 1);

        return *this;
    }

    ~Fen() { delete[] m_fen; }

   private:
    int   m_size;
    char* m_fen;
};

struct FenBatch {
    FenBatch(const std::vector<TrainingDataEntry>& entries) :
        m_size(entries.size()),
        m_fens(new Fen[entries.size()]) {
        for (int i = 0; i < m_size; ++i)
        {
            m_fens[i] = entries[i].pos.fen();
        }
    }

    ~FenBatch() { delete[] m_fens; }

   private:
    int  m_size;
    Fen* m_fens;
};

struct FenBatchStream: Stream<FenBatch> {
    static constexpr int num_feature_threads_per_reading_thread = 2;

    using BaseType = Stream<FenBatch>;

    FenBatchStream(int                                           concurrency,
                   const std::vector<std::string>&               filenames,
                   int                                           batch_size,
                   bool                                          cyclic,
                   std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(std::max(1, concurrency / num_feature_threads_per_reading_thread),
                 filenames,
                 cyclic,
                 skipPredicate),
        m_concurrency(concurrency),
        m_batch_size(batch_size) {
        m_stop_flag.store(false);

        auto worker = [this]() {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while (!m_stop_flag.load())
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
                    m_batches_not_full.wait(lock, [this]() {
                        return m_batches.size() < m_concurrency + 1 || m_stop_flag.load();
                    });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }
            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
          1, concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread));

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

    FenBatch* next() {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock,
                           [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

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

    ~FenBatchStream() {
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
    int                     m_batch_size;
    int                     m_concurrency;
    std::deque<FenBatch*>   m_batches;
    std::mutex              m_batch_mutex;
    std::mutex              m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool        m_stop_flag;
    std::atomic_int         m_num_workers;

    std::vector<std::thread> m_workers;
};

struct DataloaderSkipConfig {
    bool   filtered;
    int    random_fen_skipping;
    bool   wld_filtered;
    int    early_fen_skipping;
    int    simple_eval_skipping;
    int    param_index;
    double pc_y1, pc_y2, pc_y3;
};

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config) {
    if (config.filtered || config.random_fen_skipping || config.wld_filtered
        || config.early_fen_skipping)
    {
        return [config, prob = double(config.random_fen_skipping)
                             / (config.random_fen_skipping + 1)](const TrainingDataEntry& e) {
            // VALUE_NONE from Stockfish.
            // We need to allow a way to skip predetermined positions without
            // having to remove them from the dataset, as otherwise the we lose some
            // compression ability.
            static constexpr int VALUE_NONE = 32002;

            // lagrange interpolation weights for desired piece count distribution
            auto desired_piece_count_weights = [&config](int pc) -> double {
                double x  = pc;
                double x1 = 0, y1 = config.pc_y1;
                double x2 = 16, y2 = config.pc_y2;
                double x3 = 32, y3 = config.pc_y3;
                double l1 = (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3));
                double l2 = (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3));
                double l3 = (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2));
                return l1 * y1 + l2 * y2 + l3 * y3;
            };

            // keep stats on passing pieces
            static thread_local double alpha                            = 1;
            static thread_local double piece_count_history_all[33]      = {0};
            static thread_local double piece_count_history_passed[33]   = {0};
            static thread_local double piece_count_history_all_total    = 0;
            static thread_local double piece_count_history_passed_total = 0;

            // max skipping rate
            static constexpr double max_skipping_rate = 10.0;

            auto do_wld_skip = [&]() {
                std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto&                       prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() { return (e.isCapturingMove() || e.isInCheck()); };

            // Allow for predetermined filtering without the need to remove positions from the dataset.
            if (e.score == VALUE_NONE)
                return true;

            if (e.ply <= config.early_fen_skipping)
                return true;

            if (config.random_fen_skipping && do_skip())
                return true;

            if (config.filtered && do_filter())
                return true;

            if (config.wld_filtered && do_wld_skip())
                return true;

            if (config.simple_eval_skipping > 0
                && std::abs(e.pos.simple_eval()) < config.simple_eval_skipping)
                return true;

            constexpr bool do_debug_print = false;
            if (do_debug_print)
            {
                if (uint64_t(piece_count_history_all_total) % 10000 == 0)
                {
                    std::cout << "Total : " << piece_count_history_all_total << '\n';
                    std::cout << "Passed: " << piece_count_history_passed_total << '\n';
                    for (int i = 0; i < 33; ++i)
                        std::cout << i << ' ' << piece_count_history_passed[i] << '\n';
                }
            }

            const int pc = e.pos.piecesBB().count();
            piece_count_history_all[pc] += 1;
            piece_count_history_all_total += 1;

            double desired_piece_count_weights_total = [&desired_piece_count_weights]() {
                double tot = 0;
                for (int i = 0; i < 33; i++)
                    tot += desired_piece_count_weights(i);
                return tot;
            }();

            // update alpha, which scales the filtering probability, to a maximum rate.
            if (uint64_t(piece_count_history_all_total) % 10000 == 0)
            {
                double pass = piece_count_history_all_total * desired_piece_count_weights_total;
                for (int i = 0; i < 33; ++i)
                {
                    if (desired_piece_count_weights(pc) > 0)
                    {
                        double tmp =
                          piece_count_history_all_total * desired_piece_count_weights(pc)
                          / (desired_piece_count_weights_total * piece_count_history_all[pc]);
                        if (tmp < pass)
                            pass = tmp;
                    }
                }
                alpha = 1.0 / (pass * max_skipping_rate);
            }

            double tmp = alpha * piece_count_history_all_total * desired_piece_count_weights(pc)
                       / (desired_piece_count_weights_total * piece_count_history_all[pc]);
            tmp = std::min(1.0, tmp);
            std::bernoulli_distribution distrib(1.0 - tmp);
            auto&                       prng = rng::get_thread_local_rng();
            if (distrib(prng))
                return true;

            piece_count_history_passed[pc] += 1;
            piece_count_history_passed_total += 1;

            return false;
        };
    }

    return nullptr;
}

extern "C" {

EXPORT SparseBatch* get_sparse_batch_from_fens(const char*        feature_set_c,
                                               int                num_fens,
                                               const char* const* fens,
                                               int*               scores,
                                               int*               plies,
                                               int*               results) {
    std::vector<TrainingDataEntry> entries;
    entries.reserve(num_fens);
    for (int i = 0; i < num_fens; ++i)
    {
        auto& e = entries.emplace_back();
        e.pos   = Position::fromFen(fens[i]);
        movegen::forEachLegalMove(e.pos, [&](Move m) { e.move = m; });
        e.score  = scores[i];
        e.ply    = plies[i];
        e.result = results[i];
    }

    auto feature_variant = get_feature(feature_set_c);

    return std::visit(
      [&](const auto fs) -> SparseBatch* {
          using F = std::decay_t<decltype(fs)>;
          if constexpr (std::is_same_v<F, std::monostate>)
          {
              return nullptr;
          }
          else
          {
              return new SparseBatch(FeatureSet<decltype(fs)>{}, entries);
          }
      },
      feature_variant);
}

// changing the signature needs matching changes in data_loader/_native.py
EXPORT FenBatchStream* CDECL create_fen_batch_stream(int                  concurrency,
                                                     int                  num_files,
                                                     const char* const*   filenames,
                                                     int                  batch_size,
                                                     bool                 cyclic,
                                                     DataloaderSkipConfig config) {
    auto skipPredicate = make_skip_predicate(config);
    auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

    return new FenBatchStream(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
}

EXPORT void CDECL destroy_fen_batch_stream(FenBatchStream* stream) { delete stream; }

// changing the signature needs matching changes in data_loader/_native.py
EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char*          feature_set_c,
                                                             int                  concurrency,
                                                             int                  num_files,
                                                             const char* const*   filenames,
                                                             int                  batch_size,
                                                             bool                 cyclic,
                                                             DataloaderSkipConfig config) {
    auto skipPredicate = make_skip_predicate(config);
    auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

    auto feature_variant = get_feature(feature_set_c);

    return std::visit(
      [&](const auto fs) -> Stream<SparseBatch>* {
          using F = std::decay_t<decltype(fs)>;
          if constexpr (std::is_same_v<F, std::monostate>)
          {
              return nullptr;
          }
          else
          {
              return new FeaturedBatchStream<FeatureSet<decltype(fs)>, SparseBatch>(
                concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
          }
      },
      feature_variant);
}

EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream) { delete stream; }

EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream) {
    return stream->next();
}

EXPORT FenBatch* CDECL fetch_next_fen_batch(Stream<FenBatch>* stream) { return stream->next(); }

EXPORT void CDECL destroy_sparse_batch(SparseBatch* e) { delete e; }

EXPORT void CDECL destroy_fen_batch(FenBatch* e) { delete e; }
}

#if defined(BENCH)

/* benches
   compile and run with:
     g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native training_data_loader.cpp && ./a.out /path/to/binpack
*/

    #include <chrono>
    #include <iostream>
    #include <iomanip>
    #include <fstream>
    #include <string>

long long get_rchar_self() {
    std::ifstream io_file("/proc/self/io");
    std::string   line;
    while (std::getline(io_file, line))
    {
        if (line.rfind("rchar:", 0) == 0)
        {
            return std::stoll(line.substr(6));
        }
    }
    return -1;  // Error or not found
}

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " file1 [file2 ...]\n";
        return 1;
    }
    const char** files      = const_cast<const char**>(&argv[1]);
    int          file_count = argc - 1;

    #ifdef PGO_BUILD
    const int concurrency = 1;
    #else
    const int concurrency = std::thread::hardware_concurrency();
    #endif
    // some typical numbers, more skipping means more load
    const int                  batch_size = 16384;
    const bool                 cyclic     = true;
    const DataloaderSkipConfig config     = {.filtered             = true,
                                             .random_fen_skipping  = 3,
                                             .wld_filtered         = true,
                                             .early_fen_skipping   = 5,
                                             .simple_eval_skipping = 0,
                                             .param_index          = 0,
                                             .pc_y1                = 1.0,
                                             .pc_y2                = 2.0,
                                             .pc_y3                = 1.0};
    auto stream = create_sparse_batch_stream("Full_Threats", concurrency, file_count, files,
                                             batch_size, cyclic, config);

    auto t0 = std::chrono::high_resolution_clock::now();

    #ifdef PGO_BUILD
    constexpr int iteration_count = 30;
    #else
    constexpr int iteration_count = 6000;
    #endif

    for (int i = 1; i <= iteration_count; ++i)
    {
        destroy_sparse_batch(stream->next());
        auto t1 = std::chrono::high_resolution_clock::now();
        if (i % 1 == 0)
        {
            double    sec   = (t1 - t0).count() / 1e9;
            long long bytes = get_rchar_self();

            double mpos = i * batch_size / (sec * 1000 * 1000);
            double its  = i / sec;
            double mbps = bytes / (sec * 1024 * 1024);
            double bpos = bytes / (i * batch_size);

            std::cout << "\rIter: " << std::setw(8) << i                                //
                      << "   Time(s): " << std::setw(8) << std::setprecision(3) << sec  //
                      << "   MPos/s: " << std::setw(8) << std::setprecision(3) << mpos  //
                      << "   It/s: " << std::setw(8) << std::setprecision(1) << its     //
                      << "   MB/s: " << std::setw(8) << std::setprecision(1) << mbps    //
                      << "   B/pos: " << std::setw(8) << std::setprecision(1) << bpos   //
                      << std::flush;
        }
    }
    std::cout << std::endl;

    return 0;
}

#endif
