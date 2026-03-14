#include "training_data_loader_internal.h"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <future>
#include <random>
#include <cstring>
#include <cmath>

#include "lib/rng.h"

using namespace binpack;
using namespace chess;

// ---------------------------------------------------------
// Internal extractors and threat arrays
// ---------------------------------------------------------

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

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color) {
        auto& pos    = e.pos;
        auto  pieces = pos.piecesBB();
        auto  ksq    = pos.kingSquare(color);

        int j = 0;
        for (Square sq : pieces) {
            auto p      = pos.pieceAt(sq);
            values[j]   = 1.0f;
            features[j] = feature_index(color, ksq, sq, p);
            ++j;
        }
        return {j, INPUTS};
    }
};

struct HalfKAv2_hmExtractor: IFeatureExtractor {
    int inputs() const override { return HalfKAv2_hm::INPUTS; }
    int max_active_features() const override { return HalfKAv2_hm::MAX_ACTIVE_FEATURES; }
    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int*                     features,
                                             float*                   values,
                                             Color                    color) const override {
        return HalfKAv2_hm::fill_features_sparse(e, features, values, color);
    }
};

constexpr int numvalidtargets[12] = {6, 6, 10, 10, 8, 8, 8, 8, 10, 10, 0, 0};

using ThreatOffsetTable = std::array<std::array<int, 66>, 12>;

struct ThreatFeatureCalculation {
    ThreatOffsetTable table;
    int               totalfeatures;
};

constexpr auto threatfeaturecalc = []() {
    ThreatOffsetTable t{};

    constexpr auto pseudo_attacks = bb::detail::generatePseudoAttacks();
    int            pieceoffset    = 0;

    Piece piecetbl[12] = {whitePawn, blackPawn, whiteKnight, blackKnight, whiteBishop, blackBishop,
                          whiteRook, blackRook, whiteQueen,  blackQueen,  whiteKing,   blackKing};

    for (int c = 0; c < 2; c++) {
        for (int pt = 0; pt < 6; pt++) {
            int piece        = 2 * pt + c;
            t[piece][65]     = pieceoffset;
            int squareoffset = 0;
            for (int from = (int) a1; from <= (int) h8; from++) {
                t[piece][from] = squareoffset;
                if (piecetbl[piece].type() != PieceType::Pawn) {
                    Bitboard attacks = pseudo_attacks[piecetbl[piece].type()][Square(from)];
                    squareoffset += attacks.count();
                }
                else if (from >= (int) a2 && from <= (int) h7) {
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

constexpr ThreatOffsetTable threatoffsets  = threatfeaturecalc.table;
constexpr int               threatfeatures = threatfeaturecalc.totalfeatures;
static_assert(threatfeatures == 60144);

struct FullThreats {
    static constexpr std::string_view NAME = "Full_Threats";

    static constexpr int SQUARE_NB           = 64;
    static constexpr int PIECE_NB            = 12;
    static constexpr int COLOR_NB            = 2;
    static constexpr int PIECE_TYPE_NB       = 6;
    static constexpr int MAX_ACTIVE_FEATURES = 128;

    static constexpr int INPUTS = threatfeatures;  // 60,144

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
    // clang-format on

    static int
    threat_index(Color Perspective, Piece attkr, Square from, Square to, Piece attkd, Square ksq) {
        bool enemy = (attkr.color() != attkd.color());
        from       = (Square) (int(from) ^ (int) OrientTBL[(int) Perspective][(int) ksq]);
        to         = (Square) (int(to) ^ (int) OrientTBL[(int) Perspective][(int) ksq]);
        if (Perspective == Color::Black) {
            attkr = Piece::fromId((int) attkr ^ 1);
            attkd = Piece::fromId((int) attkd ^ 1);
        }
        if ((map[(int) attkr.type()][(int) attkd.type()] < 0)
            || (attkr.type() == attkd.type() && (enemy || attkr.type() != PieceType::Pawn)
                && from < to)) {
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
        for (int i = (int) Color::White; i <= (int) Color::Black; i++) {
            for (int j = (int) PieceType::Pawn; j <= (int) PieceType::King; j++) {
                Color     c     = order[(int) color][i];
                PieceType pt    = PieceType(j);
                Piece     attkr = Piece(pt, c);
                Bitboard  bb    = pos.piecesBB(attkr);
                if (pt == PieceType::Pawn) {
                    auto right         = (c == Color::White) ? Offset(1, 1) : Offset(-1, -1);
                    auto left          = (c == Color::White) ? Offset(-1, 1) : Offset(1, -1);
                    auto attacks_left  = bb.shifted(right) & pieces;
                    auto attacks_right = bb.shifted(left) & pieces;
                    for (Square to : attacks_left) {
                        Square from  = Square((int) to - (c == Color::White ? 9 : -9));
                        Piece  attkd = pos.pieceAt(to);
                        int    index = threat_index(color, attkr, from, to, attkd, ksq);
                        if (index >= 0) {
                            values[k]   = 1.0f;
                            features[k] = index;
                            k++;
                        }
                    }
                    for (Square to : attacks_right) {
                        Square from  = Square((int) to - (c == Color::White ? 7 : -7));
                        Piece  attkd = pos.pieceAt(to);
                        int    index = threat_index(color, attkr, from, to, attkd, ksq);
                        if (index >= 0) {
                            values[k]   = 1.0f;
                            features[k] = index;
                            k++;
                        }
                    }
                }
                else {
                    for (Square from : bb) {
                        Bitboard attacks = pos.attacks(from) & pieces;
                        for (Square to : attacks) {
                            Piece attkd = pos.pieceAt(to);
                            int   index = threat_index(color, attkr, from, to, attkd, ksq);
                            if (index >= 0) {
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

struct FullThreatsExtractor: IFeatureExtractor {
    int inputs() const override { return FullThreats::INPUTS; }
    int max_active_features() const override { return FullThreats::MAX_ACTIVE_FEATURES; }
    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int*                     features,
                                             float*                   values,
                                             Color                    color) const override {
        return FullThreats::fill_features_sparse(e, features, values, color);
    }
};

struct ComposedFeatureExtractor: IFeatureExtractor {
    std::vector<std::unique_ptr<IFeatureExtractor>> extractors;
    int                                             m_inputs;
    int                                             m_max_active;

    ComposedFeatureExtractor(std::vector<std::unique_ptr<IFeatureExtractor>> exts) :
        extractors(std::move(exts)),
        m_inputs(0),
        m_max_active(0) {
        for (auto& e : extractors)
        {
            m_inputs += e->inputs();
            m_max_active += e->max_active_features();
        }
    }

    int inputs() const override { return m_inputs; }
    int max_active_features() const override { return m_max_active; }

    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int*                     features,
                                             float*                   values,
                                             Color                    color) const override {
        int total_written = 0;
        int input_offset  = 0;

        for (auto& ext : extractors)
        {
            auto [written, ext_inputs] =
              ext->fill_features_sparse(e, features + total_written, values + total_written, color);

            // Offset the feature indices for this component
            for (int i = 0; i < written; ++i)
                features[total_written + i] += input_offset;

            input_offset += ext_inputs;
            total_written += written;
        }

        return {total_written, m_inputs};
    }
};

static std::unique_ptr<IFeatureExtractor> make_single_extractor(std::string_view name) {
    if (name == "HalfKAv2_hm")
        return std::make_unique<HalfKAv2_hmExtractor>();
    if (name == "Full_Threats")
        return std::make_unique<FullThreatsExtractor>();
    return nullptr;
}

std::shared_ptr<IFeatureExtractor> get_feature(std::string_view name) {
    std::vector<std::unique_ptr<IFeatureExtractor>> components;
    std::size_t                                     start = 0;

    while (start < name.size())
    {
        auto pos  = name.find('+', start);
        auto part = name.substr(start, pos == std::string_view::npos ? pos : pos - start);
        auto ext  = make_single_extractor(part);

        if (!ext)
        {
            std::cerr << "Unknown feature component: " << part << std::endl;
            return nullptr;
        }

        components.push_back(std::move(ext));
        start = (pos == std::string_view::npos) ? name.size() : pos + 1;
    }

    if (components.empty())
        return nullptr;

    if (components.size() == 1)
        return std::shared_ptr<IFeatureExtractor>(std::move(components[0]));

    return std::make_shared<ComposedFeatureExtractor>(std::move(components));
}

// ---------------------------------------------------------
// Class Implementations
// ---------------------------------------------------------

SparseBatch::SparseBatch(const IFeatureExtractor&              feature_set,
                         const std::vector<TrainingDataEntry>& entries) {
    num_inputs          = feature_set.inputs();
    size                = entries.size();
    max_active_features = feature_set.max_active_features();
    is_white            = new float[size];
    outcome             = new float[size];
    score               = new float[size];
    white               = new int[size * max_active_features];
    black               = new int[size * max_active_features];
    white_values        = new float[size * max_active_features];
    black_values        = new float[size * max_active_features];
    psqt_indices        = new int[size];
    layer_stack_indices = new int[size];

    num_active_white_features = 0;
    num_active_black_features = 0;

        for (int i = 0; i < size * max_active_features; ++i)
            white[i] = -1;
        for (int i = 0; i < size * max_active_features; ++i)
            black[i] = -1;
        for (int i = 0; i < size * max_active_features; ++i)
            white_values[i] = 0.0f;
        for (int i = 0; i < size * max_active_features; ++i)
            black_values[i] = 0.0f;

        for (int i = 0; i < size; ++i)
            fill_entry(feature_set, i, entries[i]);
}

SparseBatch::~SparseBatch() {
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

void SparseBatch::fill_entry(const IFeatureExtractor& fs, int i, const TrainingDataEntry& e) {
    is_white[i]            = static_cast<float>(e.pos.sideToMove() == Color::White);
    outcome[i]             = (e.result + 1.0f) / 2.0f;
    score[i]               = e.score;
    psqt_indices[i]        = (e.pos.piecesBB().count() - 1) / 4;
    layer_stack_indices[i] = psqt_indices[i];
    fill_features(fs, i, e);
}

void SparseBatch::fill_features(const IFeatureExtractor& fs, int i, const TrainingDataEntry& e) {
    const int offset = i * max_active_features;
        num_active_white_features +=
          fs.fill_features_sparse(e, white + offset, white_values + offset, Color::White).first;
        num_active_black_features +=
          fs.fill_features_sparse(e, black + offset, black_values + offset, Color::Black).first;
}

int FeaturedBatchStream::calculate_num_reader_threads(int concurrency) {
        if (worker_thread_ratio >= 1) return 1;
        return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FeaturedBatchStream::calculate_num_worker_threads(int concurrency) {
        if (worker_thread_ratio <= 0) return 1;
        return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FeaturedBatchStream::FeaturedBatchStream(std::shared_ptr<IFeatureExtractor> feature_set,
                                         int concurrency,
                                         const std::vector<std::string>& filenames,
                                         int batch_size,
                                         bool cyclic,
                                         std::function<bool(const TrainingDataEntry&)> skipPredicate,
                                         int rank,
                                         int world_size) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames, cyclic, skipPredicate, rank, world_size),
    m_feature_set(std::move(feature_set)),
    m_concurrency(concurrency),
    m_batch_size(batch_size),
    m_num_workers(calculate_num_worker_threads(concurrency)) {

    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load()) {
            entries.clear();
            {
                BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
                if (entries.empty()) break;
            }

            auto batch = new SparseBatch(*m_feature_set, entries);

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

    const int num_worker_threads = calculate_num_worker_threads(concurrency);
    for (int i = 0; i < num_worker_threads; ++i) {
        m_workers.emplace_back(worker);
    }
}

FeaturedBatchStream::~FeaturedBatchStream() {
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers) {
        if (worker.joinable()) worker.join();
    }
    for (auto& batch : m_batches) delete batch;
}

SparseBatch* FeaturedBatchStream::next() {
    std::unique_lock lock(m_batch_mutex);
    m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });
    if (!m_batches.empty()) {
        auto batch = m_batches.front();
        m_batches.pop_front();
        lock.unlock();
        m_batches_not_full.notify_one();
        return batch;
    }
    return nullptr;
}

Fen::Fen() : m_fen(nullptr) {}

Fen::Fen(const std::string& fen) : m_size(fen.size()), m_fen(new char[fen.size() + 1]) {
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
}

Fen& Fen::operator=(const std::string& fen) {
    if (m_fen != nullptr) delete[] m_fen;
    m_size = fen.size();
    m_fen  = new char[fen.size() + 1];
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    return *this;
}

Fen::~Fen() { delete[] m_fen; }

FenBatch::FenBatch(const std::vector<TrainingDataEntry>& entries) :
    m_size(entries.size()), m_fens(new Fen[entries.size()]) {
    for (int i = 0; i < m_size; ++i) m_fens[i] = entries[i].pos.fen();
}

FenBatch::~FenBatch() { delete[] m_fens; }

int FenBatchStream::calculate_num_reader_threads(int concurrency) {
        if (worker_thread_ratio >= 1) return 1;
        return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FenBatchStream::calculate_num_worker_threads(int concurrency) {
        if (worker_thread_ratio <= 0) return 1;
        return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FenBatchStream::FenBatchStream(int concurrency,
                               const std::vector<std::string>& filenames,
                               int batch_size,
                               bool cyclic,
                               std::function<bool(const TrainingDataEntry&)> skipPredicate,
                               int rank,
                               int world_size) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames, cyclic, skipPredicate, rank, world_size),
    m_concurrency(concurrency),
    m_batch_size(batch_size),
    m_num_workers(calculate_num_worker_threads(concurrency)) {

    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load()) {
            entries.clear();
            {
                BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
                if (entries.empty()) break;
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

    const int num_worker_threads = calculate_num_worker_threads(concurrency);
    for (int i = 0; i < num_worker_threads; ++i) {
        m_workers.emplace_back(worker);
    }
}

FenBatchStream::~FenBatchStream() {
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers) {
        if (worker.joinable()) worker.join();
    }
    for (auto& batch : m_batches) delete batch;
}

FenBatch* FenBatchStream::next() {
    std::unique_lock lock(m_batch_mutex);
    m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });
    if (!m_batches.empty()) {
        auto batch = m_batches.front();
        m_batches.pop_front();
        lock.unlock();
        m_batches_not_full.notify_one();
        return batch;
    }
    return nullptr;
}

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config) {
    if (config.filtered || config.random_fen_skipping || config.wld_filtered || config.early_fen_skipping) {
        return [config, prob = double(config.random_fen_skipping) / (config.random_fen_skipping + 1)](const TrainingDataEntry& e) {
            static constexpr int VALUE_NONE = 32002;

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

            static thread_local double alpha                            = 1;
            static thread_local double piece_count_history_all[33]      = {0};
            static thread_local double piece_count_history_passed[33]   = {0};
            static thread_local double piece_count_history_all_total    = 0;
            static thread_local double piece_count_history_passed_total = 0;

            static constexpr double max_skipping_rate = 10.0;

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

            auto do_filter = [&]() { return (e.isCapturingMove() || e.isInCheck()); };

            if (e.score == VALUE_NONE) return true;
            if (e.ply <= config.early_fen_skipping) return true;
            if (config.random_fen_skipping && do_skip()) return true;
            if (config.filtered && do_filter()) return true;
            if (config.wld_filtered && do_wld_skip()) return true;
            if (config.simple_eval_skipping > 0 && std::abs(e.pos.simple_eval()) < config.simple_eval_skipping) return true;

            const int pc = e.pos.piecesBB().count();
            piece_count_history_all[pc] += 1;
            piece_count_history_all_total += 1;

            double desired_piece_count_weights_total = [&desired_piece_count_weights]() {
                double tot = 0;
                for (int i = 0; i < 33; i++) tot += desired_piece_count_weights(i);
                return tot;
            }();

            // update alpha, which scales the filtering probability, to a maximum rate.
            if (uint64_t(piece_count_history_all_total) % 10000 == 0) {
                double pass = piece_count_history_all_total * desired_piece_count_weights_total;
                for (int i = 0; i < 33; ++i) {
                    if (desired_piece_count_weights(pc) > 0) {
                        double tmp =
                          piece_count_history_all_total * desired_piece_count_weights(pc)
                          / (desired_piece_count_weights_total * piece_count_history_all[pc]);
                        if (tmp < pass)
                        pass = tmp;
                    }
                }
                alpha = 1.0 / (pass * max_skipping_rate);
            }

            double tmp = alpha * piece_count_history_all_total * desired_piece_count_weights(pc) / (desired_piece_count_weights_total * piece_count_history_all[pc]);
            tmp = std::min(1.0, tmp);
            std::bernoulli_distribution distrib(1.0 - tmp);
            auto& prng = rng::get_thread_local_rng();
            if (distrib(prng)) return true;

            piece_count_history_passed[pc] += 1;
            piece_count_history_passed_total += 1;

            return false;
        };
    }
    return nullptr;
}
