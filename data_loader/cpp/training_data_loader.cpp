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

// Ply that marks search-tree samples: standalone positions without a game.
static constexpr int TREE_SAMPLE_PLY = 9999;

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

struct K32H3 {
    static constexpr std::string_view NAME = "K32H3";

    static constexpr int NUM_SQ              = 64;
    static constexpr int NUM_PT              = 12;
    static constexpr int NUM_PLANES          = NUM_SQ * NUM_PT;
    static constexpr int NUM_KING_BUCKETS    = NUM_SQ / 2;
    static constexpr int NUM_STATES          = 3;
    static constexpr int NUM_BUCKETS         = NUM_KING_BUCKETS * NUM_STATES;
    static constexpr int INPUTS              = NUM_PLANES * NUM_BUCKETS;
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

    static int block_idx(const Position& pos, Color color) {
        if (pos.piecesBB(Piece(PieceType::Queen, !color)).any())
            return 2;
        for (Color c : {Color::White, Color::Black})
            if (pos.piecesBB(Piece(PieceType::Rook, c)).any()
                || pos.piecesBB(Piece(PieceType::Queen, c)).any())
                return 1;
        return 0;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto&      pos    = e.pos;
        const auto ksq    = pos.kingSquare(color);
        const int  kb     = KingBuckets[static_cast<int>(orient_flip_2(color, ksq, ksq))];
        const int  bucket = block_idx(pos, color) * NUM_KING_BUCKETS + kb;

        int j = 0;
        for (Square sq : pos.piecesBB())
        {
            const Piece p     = pos.pieceAt(sq);
            const int   p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            features[j++]     = static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ
                          + bucket * NUM_PLANES;
        }
        return {j, INPUTS};
    }
};

// Pawn attackers no longer target pawns; those relationships are represented by PP_3Wide.
constexpr int numvalidtargets[6] = {4, 10, 8, 8, 10, 0};

constexpr int threat_piece(Piece piece) { return (int) piece.color() * 5 + (int) piece.type(); }

constexpr std::uint64_t pawn_pair_mask(int sq) {
    constexpr std::uint64_t FileA = 0x0101010101010101ULL;
    constexpr std::uint64_t Rank1 = 0xFFULL;
    constexpr std::uint64_t Rank8 = Rank1 << 56;

    const int     f     = sq & 7;
    std::uint64_t files = FileA << f;
    if (f > 0)
        files |= FileA << (f - 1);
    if (f < 7)
        files |= FileA << (f + 1);
    return files & ~Rank1 & ~Rank8 & ~(1ULL << sq);
}

using ThreatOffsetTable = std::array<std::array<int, 66>, 10>;

struct ThreatFeatureCalculation {
    ThreatOffsetTable table;
    int               totalfeatures;
};

constexpr auto threatfeaturecalc = []() {
    ThreatOffsetTable t{};

    constexpr auto pseudo_attacks = bb::detail::generatePseudoAttacks();
    int            pieceoffset    = 0;

    Piece piecetbl[10] = {whitePawn, whiteKnight, whiteBishop, whiteRook, whiteQueen,
                          blackPawn, blackKnight, blackBishop, blackRook, blackQueen};

    for (int piece = 0; piece < 10; ++piece)
    {
        t[piece][65]     = pieceoffset;
        int squareoffset = 0;

        for (int from = (int) a1; from <= (int) h8; ++from)
        {
            t[piece][from] = squareoffset;
            if (piecetbl[piece].type() == PieceType::Pawn)
            {
                if (from >= (int) a2 && from <= (int) h7)
                    squareoffset +=
                      bb::pawnAttacks(Bitboard::square(Square(from)), piecetbl[piece].color())
                        .count();
            }
            else
            {
                squareoffset += pseudo_attacks[piecetbl[piece].type()][Square(from)].count();
            }
        }

        t[piece][64] = squareoffset;
        pieceoffset += numvalidtargets[(int) piecetbl[piece].type()] * squareoffset;
    }
    return ThreatFeatureCalculation{t, pieceoffset};
}();

constexpr ThreatOffsetTable threatoffsets  = threatfeaturecalc.table;
constexpr int               threatfeatures = threatfeaturecalc.totalfeatures;
static_assert(threatfeatures == 59808);

struct FullThreats {
    static constexpr std::string_view NAME = "Full_Threats";

    static constexpr int SQUARE_NB           = 64;
    static constexpr int PIECE_NB            = 12;
    static constexpr int COLOR_NB            = 2;
    static constexpr int PIECE_TYPE_NB       = 6;
    static constexpr int MAX_ACTIVE_FEATURES = 128;

    static constexpr int INPUTS = threatfeatures;

    // clang-format off
    static constexpr Square OrientTBL[COLOR_NB][SQUARE_NB] = {
      { h1, h1, h1, h1, a1, a1, a1, a1,
        h1, h1, h1, h1, a1, a1, a1, a1,
        h1, h1, h1, h1, a1, a1, a1, a1,
        h1, h1, h1, h1, a1, a1, a1, a1,
        h1, h1, h1, h1, a1, a1, a1, a1,
        h1, h1, h1, h1, a1, a1, a1, a1,
        h1, h1, h1, h1, a1, a1, a1, a1,
        h1, h1, h1, h1, a1, a1, a1, a1 },
      { h8, h8, h8, h8, a8, a8, a8, a8,
        h8, h8, h8, h8, a8, a8, a8, a8,
        h8, h8, h8, h8, a8, a8, a8, a8,
        h8, h8, h8, h8, a8, a8, a8, a8,
        h8, h8, h8, h8, a8, a8, a8, a8,
        h8, h8, h8, h8, a8, a8, a8, a8,
        h8, h8, h8, h8, a8, a8, a8, a8,
        h8, h8, h8, h8, a8, a8, a8, a8 }
    };

    static constexpr int map[PIECE_TYPE_NB][PIECE_TYPE_NB] = {
      {-1, 0, -1, 1, -1, -1},
      { 0, 1,  2, 3,  4, -1},
      { 0, 1,  2, 3, -1, -1},
      { 0, 1,  2, 3, -1, -1},
      { 0, 1,  2, 3,  4, -1},
      {-1,-1, -1,-1, -1, -1}
    };

    // clang-format on

    static int
    threat_index(Color perspective, Piece attkr, Square from, Square to, Piece attkd, Square ksq) {
        from = (Square) ((int) from ^ (int) OrientTBL[(int) perspective][(int) ksq]);
        to   = (Square) ((int) to ^ (int) OrientTBL[(int) perspective][(int) ksq]);

        if (perspective == Color::Black)
        {
            attkr = Piece::fromId((int) attkr ^ 1);
            attkd = Piece::fromId((int) attkd ^ 1);
        }

        int slot = map[(int) attkr.type()][(int) attkd.type()];
        if (slot < 0)
            return -1;

        bool enemy = attkr.color() != attkd.color();
        if (attkr.type() == attkd.type() && (enemy || attkr.type() != PieceType::Pawn) && from < to)
            return -1;

        Bitboard attacks  = attkr.type() == PieceType::Pawn
                            ? bb::pawnAttacks(Bitboard::square(from), attkr.color())
                            : bb::detail::pseudoAttacks()[attkr.type()][from];
        int      attacker = threat_piece(attkr);
        return int(threatoffsets[attacker][65]
                   + (int(attkd.color()) * (numvalidtargets[(int) attkr.type()] / 2) + slot)
                       * threatoffsets[attacker][64]
                   + threatoffsets[attacker][(int) from]
                   + (Bitboard::fromBits((1ULL << (int) to) - 1) & attacks).count());
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos    = e.pos;
        auto  pieces = pos.piecesBB();
        // Kings are never targets; exclude them upfront
        auto  piecesNoK   = pieces & ~pos.piecesBB(whiteKing) & ~pos.piecesBB(blackKing);
        auto  ksq         = pos.kingSquare(color);
        Color order[2][2] = {{Color::White, Color::Black}, {Color::Black, Color::White}};
        int   k           = 0;

        for (int i = (int) Color::White; i <= (int) Color::Black; i++)
        {
            Color c = order[(int) color][i];

            // ---- Pawn attacks ----
            {
                Piece    attkr    = Piece(PieceType::Pawn, c);
                Bitboard bb       = pos.piecesBB(attkr);
                auto     allPawns = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);

                // Diagonal pawn threats exclude pawn targets (those are handled by PP_3Wide)
                auto diagTargets = piecesNoK & ~allPawns;

                auto right   = (c == Color::White) ? Offset(1, 1) : Offset(-1, -1);
                auto left    = (c == Color::White) ? Offset(-1, 1) : Offset(1, -1);
                int  r_delta = (c == Color::White) ? 9 : -9;
                int  l_delta = (c == Color::White) ? 7 : -7;

                for (Square to : bb.shifted(right) & diagTargets)
                {
                    Square from  = Square((int) to - r_delta);
                    Piece  attkd = pos.pieceAt(to);
                    int    index = threat_index(color, attkr, from, to, attkd, ksq);
                    if (index >= 0)
                    {
                        features[k++] = index;
                    }
                }
                for (Square to : bb.shifted(left) & diagTargets)
                {
                    Square from  = Square((int) to - l_delta);
                    Piece  attkd = pos.pieceAt(to);
                    int    index = threat_index(color, attkr, from, to, attkd, ksq);
                    if (index >= 0)
                    {
                        features[k++] = index;
                    }
                }
            }

            for (int j = (int) PieceType::Knight; j < (int) PieceType::King; j++)
            {
                Piece    attkr = Piece(PieceType(j), c);
                Bitboard bb    = pos.piecesBB(attkr);

                for (Square from : bb)
                {
                    Bitboard attacks = pos.attacks(from) & piecesNoK;
                    for (Square to : attacks)
                    {
                        Piece attkd = pos.pieceAt(to);
                        int   index = threat_index(color, attkr, from, to, attkd, ksq);
                        if (index >= 0)
                        {
                            features[k++] = index;
                        }
                    }
                }
            }
        }
        return {k, INPUTS};
    }
};

struct PP_3Wide {
    static constexpr std::string_view NAME = "PP_3Wide";

    static constexpr int PAWN_IDS            = 2 * 48;
    static constexpr int INPUTS              = PAWN_IDS * (PAWN_IDS - 1) / 2;
    static constexpr int MAX_ACTIVE_FEATURES = 100;

    static int make_pawn_id(Color color, Square square) {
        return 48 * static_cast<int>(color) + static_cast<int>(square) - static_cast<int>(a2);
    }

    static int make_index(
      Color perspective, Color color, Square from, Square to, Color paired_color, Square ksq) {
        int orient = static_cast<int>(
          FullThreats::OrientTBL[static_cast<int>(perspective)][static_cast<int>(ksq)]);
        Square from_oriented = static_cast<Square>(static_cast<int>(from) ^ orient);
        Square to_oriented   = static_cast<Square>(static_cast<int>(to) ^ orient);

        Color color_oriented =
          static_cast<Color>(static_cast<int>(color) ^ static_cast<int>(perspective));
        Color paired_color_oriented =
          static_cast<Color>(static_cast<int>(paired_color) ^ static_cast<int>(perspective));

        if (from_oriented < a2 || from_oriented > h7 || to_oriented < a2 || to_oriented > h7)
            return INPUTS;

        int id_a = make_pawn_id(color_oriented, from_oriented);
        int id_b = make_pawn_id(paired_color_oriented, to_oriented);
        int hi   = std::max(id_a, id_b);
        int lo   = std::min(id_a, id_b);

        return hi * (hi - 1) / 2 + lo;
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) {
        auto& pos         = e.pos;
        auto  allPawns    = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
        auto  ksq         = pos.kingSquare(color);
        Color order[2][2] = {{Color::White, Color::Black}, {Color::Black, Color::White}};
        int   k           = 0;

        for (int i = static_cast<int>(Color::White); i <= static_cast<int>(Color::Black); ++i)
        {
            Color    c  = order[static_cast<int>(color)][i];
            Piece    p  = Piece(PieceType::Pawn, c);
            Bitboard bb = pos.piecesBB(p);

            for (Square from : bb)
            {
                Bitboard targets =
                  Bitboard::fromBits(pawn_pair_mask(static_cast<int>(from))) & allPawns;
                for (Square to : targets)
                {
                    if (from < to)
                        continue;

                    Color paired_color = pos.pieceAt(to).color();
                    int   index        = make_index(color, c, from, to, paired_color, ksq);
                    if (index < INPUTS)
                    {
                        features[k] = index;
                        ++k;
                    }
                }
            }
        }

        return {k, INPUTS};
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

    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        int total_written = 0;
        int input_offset  = 0;

        for (auto& ext : extractors)
        {
            auto [written, ext_inputs] =
              ext->fill_features_sparse(e, features + total_written, color);

            // Offset the feature indices for this component
            for (int i = 0; i < written; ++i)
                features[total_written + i] += input_offset;

            input_offset += ext_inputs;
            total_written += written;
        }

        return {total_written, m_inputs};
    }
};

template<typename T>
struct SimpleExtractor: IFeatureExtractor {
    int inputs() const override { return T::INPUTS; }
    int max_active_features() const override { return T::MAX_ACTIVE_FEATURES; }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color color) const override {
        return T::fill_features_sparse(e, features, color);
    }
};

static std::unique_ptr<IFeatureExtractor> make_single_extractor(std::string_view name) {
    if (name == "Full_Threats")
        return std::make_unique<SimpleExtractor<FullThreats>>();
    if (name == "PP_3Wide")
        return std::make_unique<SimpleExtractor<PP_3Wide>>();
    if (name == "K32H3")
        return std::make_unique<SimpleExtractor<K32H3>>();
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

// Piece-count band: (pieces - 1) / 4
static int piece_count(const Position& pos) { return (pos.piecesBB().count() - 1) / 4; }

// 0 if no queens on the board
// 1 if only the non-side-to-move has a queen
// 2 if only the side-to-move has a queen
// 3 if both sides have a queen
static int queens(const Position& pos) {
    const Color stm = pos.sideToMove();
    return 2 * pos.piecesBB(Piece(PieceType::Queen, stm)).any()
         + pos.piecesBB(Piece(PieceType::Queen, !stm)).any();
}

// 0 if pawns stand on both board halves (files a-d and e-h), 1 if on one half only, 2 if none
static int pawn_halves(const Position& pos) {
    const Bitboard pawns     = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn);
    const Bitboard queenSide = Bitboard::betweenFiles(fileA, fileD);
    return !pawns.any() ? 2 : (pawns & queenSide).any() && (pawns & ~queenSide).any() ? 0 : 1;
}

// 1 if some side has two or more officers (knights, bishops, rooks, queens)
static int many_officers(const Position& pos) {
    const Bitboard pawnsKings = pos.piecesBB(whitePawn) | pos.piecesBB(blackPawn)
                              | pos.piecesBB(whiteKing) | pos.piecesBB(blackKing);
    const auto officers = [&](Color c) { return (pos.piecesBB(c) & ~pawnsKings).count(); };
    return std::max(officers(Color::White), officers(Color::Black)) >= 2;
}

int LayerStackKeys::index(const Position& pos) const {
    int index = 0;
    for (const auto& key : keys)
        index = index * key.states + key.value(pos);
    return index;
}

std::optional<LayerStackKeys> get_layer_stack_keys(std::string_view names) {
    LayerStackKeys result;
    std::size_t    start = 0;

    while (start < names.size())
    {
        auto pos  = names.find(':', start);
        auto name = names.substr(start, pos == std::string_view::npos ? pos : pos - start);

        if (name == "piececount")
            result.keys.push_back({8, piece_count});
        else if (name == "queen")
            result.keys.push_back({4, queens});
        else if (name == "pawnhalf")
            result.keys.push_back({3, pawn_halves});
        else if (name == "manyoffi")
            result.keys.push_back({2, many_officers});
        else
        {
            std::cerr << "Unknown layer-stack key: " << name << std::endl;
            return std::nullopt;
        }

        start = (pos == std::string_view::npos) ? names.size() : pos + 1;
    }

    return result;
}

// ---------------------------------------------------------
// Class Implementations
// ---------------------------------------------------------

SparseBatchBufferPool::SparseBatchBufferPool(std::size_t floats,
                                             std::size_t ints,
                                             std::size_t capacity) :
    m_floats(floats),
    m_ints(ints),
    m_capacity(capacity) {
    m_free.reserve(capacity);
}

SparseBatchBufferPool::Buffers SparseBatchBufferPool::acquire(std::size_t floats,
                                                              std::size_t ints) {
    if (floats == m_floats && ints == m_ints)
    {
        std::lock_guard lock(m_mutex);
        if (!m_free.empty())
        {
            auto buffers = std::move(m_free.back());
            m_free.pop_back();
            return buffers;
        }
    }
    // Uninitialized: only the part a batch fills is ever touched (and resident).
    return {std::unique_ptr<float[]>(new float[floats]), std::unique_ptr<int[]>(new int[ints])};
}

void SparseBatchBufferPool::release(Buffers buffers, std::size_t floats, std::size_t ints) {
    if (floats == m_floats && ints == m_ints)
    {
        std::lock_guard lock(m_mutex);
        if (m_free.size() < m_capacity)
            m_free.push_back(std::move(buffers));
    }
    // Partial final batches and excess buffers are freed normally. The cache
    // never retains storage with a different layout or beyond its capacity.
}

// The packed layout with every row at max_active_features: the counts are
// only known once the batch is built.
static std::size_t packed_capacity(std::size_t size, std::size_t max_active) {
    return 3 * size + 2 + 2 * size * max_active;
}

SparseBatch::SparseBatch(const IFeatureExtractor&               fs,
                         const LayerStackKeys&                  ls_keys,
                         const std::vector<TrainingDataEntry>&  entries,
                         std::shared_ptr<SparseBatchBufferPool> buffer_pool) :
#ifdef NNUE_LOADER_STATISTICS
    entries_copy(entries),
#endif
    m_buffer_pool(std::move(buffer_pool)) {
    num_inputs                = fs.inputs();
    size                      = entries.size();
    max_active_features       = fs.max_active_features();
    num_active_white_features = 0;
    num_active_black_features = 0;
    const std::size_t floats  = std::size_t(size) * 3;
    const std::size_t ints    = packed_capacity(size, max_active_features);

    if (m_buffer_pool)
    {
        auto buffers  = m_buffer_pool->acquire(floats, ints);
        m_float_block = buffers.first.release();
        m_int_block   = buffers.second.release();
    }
    else
    {
        m_float_block = new float[floats];
        m_int_block   = new int[ints];
    }
    is_white = m_float_block;
    outcome  = m_float_block + size;
    score    = m_float_block + 2 * size;
    packed   = m_int_block;

    // Black IDs and layer-stack indices are parked in the tail of the buffer
    // and moved behind the white IDs once the white count is known.
    int* const white_offsets = packed;
    int* const black_offsets = white_offsets + size + 1;
    int* const white_ids     = black_offsets + size + 1;
    int* const black_parked  = white_ids + std::size_t(size) * max_active_features;
    int* const stack_parked  = black_parked + std::size_t(size) * max_active_features;
    white_offsets[0] = black_offsets[0] = 0;
    for (int i = 0; i < size; ++i)
    {
        const TrainingDataEntry& e = entries[i];
        fill_targets(i, e);
        num_active_white_features +=
          fs.fill_features_sparse(e, white_ids + num_active_white_features, Color::White).first;
        num_active_black_features +=
          fs.fill_features_sparse(e, black_parked + num_active_black_features, Color::Black)
            .first;
        white_offsets[i + 1] = num_active_white_features;
        black_offsets[i + 1] = num_active_black_features;
        stack_parked[i]      = ls_keys.index(e.pos);
    }
    int* const black_ids = white_ids + num_active_white_features;
    std::memmove(black_ids, black_parked, std::size_t(num_active_black_features) * sizeof(int));
    std::memmove(black_ids + num_active_black_features, stack_parked,
                 std::size_t(size) * sizeof(int));
}

SparseBatch::~SparseBatch() {
    if (m_buffer_pool)
    {
        m_buffer_pool->release(
          {std::unique_ptr<float[]>(m_float_block), std::unique_ptr<int[]>(m_int_block)},
          std::size_t(size) * 3, packed_capacity(size, max_active_features));
    }
    else
    {
        delete[] m_float_block;
        delete[] m_int_block;
    }
}

void SparseBatch::fill_targets(int i, const TrainingDataEntry& e) {
    is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
    // Search-tree samples have no result: outcome 2 makes the loss use the score alone.
    outcome[i]  = e.ply == TREE_SAMPLE_PLY ? 2.0f : (e.result + 1.0f) / 2.0f;
    score[i]    = e.score;
}

void SparseBatch::unpack_into_padding(int* dense) const {
    const std::size_t row_ints      = std::size_t(size) * max_active_features;
    const int*        white_offsets = packed;
    const int*        black_offsets = white_offsets + size + 1;
    const int*        white_ids     = black_offsets + size + 1;
    const int*        black_ids     = white_ids + num_active_white_features;
    const int*        stacks        = black_ids + num_active_black_features;
    for (int row = 0; row < size; ++row)
    {
        const std::size_t offset = std::size_t(row) * max_active_features;
        std::copy(white_ids + white_offsets[row], white_ids + white_offsets[row + 1],
                  dense + offset);
        std::copy(black_ids + black_offsets[row], black_ids + black_offsets[row + 1],
                  dense + row_ints + offset);
    }
    std::copy(stacks, stacks + size, dense + 2 * row_ints);
}

int FeaturedBatchStream::calculate_num_reader_threads(int concurrency, int batch_size) {
    if (worker_thread_ratio(batch_size) >= 1)
        return 1;
    return std::max(1, concurrency - calculate_num_worker_threads(concurrency, batch_size));
}

int FeaturedBatchStream::calculate_num_worker_threads(int concurrency, int batch_size) {
    const double ratio = worker_thread_ratio(batch_size);
    if (ratio <= 0)
        return 1;
    return std::max(1, static_cast<int>(std::floor(concurrency * ratio)));
}

FeaturedBatchStream::FeaturedBatchStream(
  std::shared_ptr<IFeatureExtractor>            feature_set,
  LayerStackKeys                                ls_keys,
  int                                           concurrency,
  const std::vector<std::string>&               filenames,
  int                                           batch_size,
  bool                                          cyclic,
  std::function<bool(const TrainingDataEntry&)> skipPredicate,
  int                                           rank,
  int                                           world_size,
  nnue::UniquePositionCounter*                  counter,
  DataloaderIOConfig                            io_config) :
    BaseType(calculate_num_reader_threads(concurrency, batch_size),
             filenames,
             cyclic,
             skipPredicate,
             counter,
             rank,
             world_size,
             io_config,
             calculate_num_worker_threads(concurrency, batch_size)),
    m_feature_set(std::move(feature_set)),
    m_ls_keys(std::move(ls_keys)),
    m_batch_size(batch_size),
    m_concurrency(concurrency),
    m_num_workers(calculate_num_worker_threads(concurrency, batch_size)),
    m_batch_queue_capacity(calculate_num_worker_threads(concurrency, batch_size) + 4) {

    // At most one fetched batch, the finished deque, and one batch per builder
    // normally exist at a time. Recycle that working set without growing a
    // process-global cache or keeping buffers alive after the stream closes.
    m_buffer_pool = std::make_shared<SparseBatchBufferPool>(
      std::size_t(batch_size) * 3,
      packed_capacity(batch_size, m_feature_set->max_active_features()),
      std::size_t(m_batch_queue_capacity) + m_num_workers.load() + 1);
    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load())
        {
            entries.clear();
            {
                BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
                if (entries.empty())
                    break;
            }

            auto batch = new SparseBatch(*m_feature_set, m_ls_keys, entries, m_buffer_pool);

            {
                std::unique_lock lock(m_batch_mutex);
                m_batches_not_full.wait(lock, [this]() {
                    return m_batches.size() < static_cast<size_t>(m_batch_queue_capacity)
                        || m_stop_flag.load();
                });
                m_batches.emplace_back(batch);
                lock.unlock();
                m_batches_any.notify_one();
            }
        }
        m_num_workers.fetch_sub(1);
        m_batches_any.notify_one();
    };

    const int num_worker_threads = calculate_num_worker_threads(concurrency, batch_size);
    for (int i = 0; i < num_worker_threads; ++i)
    {
        m_workers.emplace_back(worker);
    }
}

FeaturedBatchStream::~FeaturedBatchStream() {
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
    for (auto& batch : m_batches)
        delete batch;
    // Counter is owned by AnyStream and deleted after m_stream (and its
    // workers) are destroyed, preventing use-after-free.
}

SparseBatch* FeaturedBatchStream::next() {
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

Fen::Fen() :
    m_fen(nullptr) {}

Fen::Fen(const std::string& fen) :
    m_size(fen.size()),
    m_fen(new char[fen.size() + 1]) {
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
}

Fen& Fen::operator=(const std::string& fen) {
    if (m_fen != nullptr)
        delete[] m_fen;
    m_size = fen.size();
    m_fen  = new char[fen.size() + 1];
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    return *this;
}

Fen::~Fen() { delete[] m_fen; }

FenBatch::FenBatch(const std::vector<TrainingDataEntry>& entries) :
    m_size(entries.size()),
    m_fens(new Fen[entries.size()]) {
    for (int i = 0; i < m_size; ++i)
        m_fens[i] = entries[i].pos.fen();
}

FenBatch::~FenBatch() { delete[] m_fens; }

int FenBatchStream::calculate_num_reader_threads(int concurrency) {
    if (worker_thread_ratio >= 1)
        return 1;
    return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FenBatchStream::calculate_num_worker_threads(int concurrency) {
    if (worker_thread_ratio <= 0)
        return 1;
    return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FenBatchStream::FenBatchStream(int                                           concurrency,
                               const std::vector<std::string>&               filenames,
                               int                                           batch_size,
                               bool                                          cyclic,
                               std::function<bool(const TrainingDataEntry&)> skipPredicate,
                               int                                           rank,
                               int                                           world_size,
                               nnue::UniquePositionCounter*                  counter,
                               DataloaderIOConfig                            io_config) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames,
             cyclic,
             skipPredicate,
             counter,
             rank,
             world_size,
             io_config,
             calculate_num_worker_threads(concurrency)),
    m_batch_size(batch_size),
    m_concurrency(concurrency),
    m_num_workers(calculate_num_worker_threads(concurrency)),
    m_batch_queue_capacity(calculate_num_worker_threads(concurrency) + 4) {

    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load())
        {
            entries.clear();
            {
                BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
                if (entries.empty())
                    break;
            }

            auto batch = new FenBatch(entries);

            {
                std::unique_lock lock(m_batch_mutex);
                m_batches_not_full.wait(lock, [this]() {
                    return m_batches.size() < static_cast<size_t>(m_batch_queue_capacity)
                        || m_stop_flag.load();
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
    for (int i = 0; i < num_worker_threads; ++i)
    {
        m_workers.emplace_back(worker);
    }
}

FenBatchStream::~FenBatchStream() {
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
    for (auto& batch : m_batches)
        delete batch;
    // Counter is owned by AnyStream and deleted after m_stream (and its
    // workers) are destroyed, preventing use-after-free.
}

FenBatch* FenBatchStream::next() {
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

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config) {
    if (!config.filtered && !config.wld_filtered && config.random_fen_skipping <= 0
        && config.early_fen_skipping < 0 && config.soft_early_fen_skipping <= 0
        && config.decided_score <= 0)
    {
        return nullptr;
    }

    double   skip_prob             = 0.0;
    uint64_t random_skip_threshold = 0;
    if (config.random_fen_skipping > 0)
    {
        skip_prob = double(config.random_fen_skipping) / (config.random_fen_skipping + 1);
        random_skip_threshold = static_cast<uint64_t>(skip_prob * static_cast<double>(~0ULL));
    }

    // --- Precompute 5-Point Spline PC LUT ---
    std::array<double, 33> target_pc_weights_lut{};
    double                 target_pc_weights_total = 0.0;

    auto desired_piece_count_weights = [&config](int pc) -> double {
        double x    = static_cast<double>(pc);
        double y[5] = {config.pc_y0, config.pc_y1, config.pc_y2, config.pc_y3, config.pc_y4};

        if (x <= 0)
            return y[0];
        if (x >= 32)
            return y[4];

        int i = static_cast<int>(x / 8.0);
        if (i > 3)
            i = 3;

        double x0 = i * 8.0;
        double t  = (x - x0) / 8.0;

        auto get_slope = [&](int idx) {
            if (idx == 0)
                return (y[1] - y[0]);
            if (idx == 4)
                return (y[4] - y[3]);
            return (y[idx + 1] - y[idx - 1]) / 2.0;
        };

        double m0 = get_slope(i);
        double m1 = get_slope(i + 1);

        double t2 = t * t;
        double t3 = t2 * t;

        double h00 = 2 * t3 - 3 * t2 + 1;
        double h10 = t3 - 2 * t2 + t;
        double h01 = -2 * t3 + 3 * t2;
        double h11 = t3 - t2;

        double val = h00 * y[i] + h10 * m0 + h01 * y[i + 1] + h11 * m1;

        return std::max(0.0, val);
    };

    for (int i = 0; i < 33; ++i)
    {
        target_pc_weights_lut[i] = desired_piece_count_weights(i);
        target_pc_weights_total += target_pc_weights_lut[i];
    }
    if (target_pc_weights_total <= 0.0)
    {
        std::fill(target_pc_weights_lut.begin(), target_pc_weights_lut.end(), 1.0);
        target_pc_weights_total = static_cast<double>(target_pc_weights_lut.size());
    }

    // --- Precompute Soft Early Ply Filter LUT ---
    std::vector<double> early_ply_accept_prob;

    if (config.soft_early_fen_skipping > 0)
    {
        size_t lut_size = static_cast<size_t>(config.soft_early_fen_skipping) + 1;
        early_ply_accept_prob.resize(lut_size);

        auto interpolate_ply = [&config](double ply) -> double {
            struct Pt {
                double x, y;
            };
            Pt pts[5] = {{config.ply_x1, config.ply_y1},
                         {config.ply_x2, config.ply_y2},
                         {config.ply_x3, config.ply_y3},
                         {config.ply_x4, config.ply_y4},
                         {static_cast<double>(config.soft_early_fen_skipping), 1.0}};

            if (ply <= pts[0].x)
                return pts[0].y;
            if (ply >= pts[4].x)
                return pts[4].y;

            for (int i = 0; i < 4; ++i)
            {
                if (ply >= pts[i].x && ply <= pts[i + 1].x)
                {
                    if (pts[i + 1].x == pts[i].x)
                        return pts[i].y;
                    double t = (ply - pts[i].x) / (pts[i + 1].x - pts[i].x);
                    return pts[i].y + t * (pts[i + 1].y - pts[i].y);
                }
            }
            return 1.0;
        };

        for (size_t i = 0; i < lut_size; ++i)
        {
            early_ply_accept_prob[i] =
              std::clamp(interpolate_ply(static_cast<double>(i)), 0.0, 1.0);
        }
    }

    const uint64_t tree_sample_skip_threshold = static_cast<uint64_t>(
      (1.0 - std::clamp(config.tree_sample_accept, 0.0, 1.0)) * static_cast<double>(~0ULL));
    const uint64_t decided_skip_threshold = static_cast<uint64_t>(
      (1.0 - std::clamp(config.decided_accept, 0.0, 1.0)) * static_cast<double>(~0ULL));

    return [config, random_skip_threshold, tree_sample_skip_threshold, decided_skip_threshold,
            target_pc_weights_lut, target_pc_weights_total,
            early_ply_accept_prob = std::move(early_ply_accept_prob)](const TrainingDataEntry& e) {
        static constexpr int    VALUE_NONE = 32002;

        if (e.ply == TREE_SAMPLE_PLY)
            return e.score == VALUE_NONE
                || rng::get_thread_local_rng()() < tree_sample_skip_threshold;

        static thread_local int last_ply   = -1;
        static thread_local int last_score = VALUE_NONE;

        // skip when score 0 was used as placeholder
        // detected through heuristic:
        // Game was not a draw and
        // last valid score was somewhat large

        bool skip_placeholder_zero = e.ply > last_ply && last_score != VALUE_NONE
                                  && std::abs(last_score) > 100 && e.result != 0 && e.score == 0;

        last_ply = e.ply;

        if (e.score == VALUE_NONE)
            return true;
        if (skip_placeholder_zero)
            return true;

        // Only update if valid score.
        last_score = e.score;

        // Hard Early Ply Filter
        if (e.ply <= config.early_fen_skipping)
            return true;

        auto& prng = rng::get_thread_local_rng();

        if (config.random_fen_skipping && (prng() < random_skip_threshold))
            return true;
        if (config.filtered && (e.isCapturingMove() || e.isInCheck()))
            return true;

        if (config.wld_filtered)
        {
            uint64_t wld_skip_threshold =
              static_cast<uint64_t>((1.0 - e.score_result_prob()) * static_cast<double>(~0ULL));
            if (prng() < wld_skip_threshold)
                return true;
        }

        if (config.decided_score > 0 && std::abs(e.score) >= config.decided_score
            && prng() < decided_skip_threshold)
            return true;

        if (config.simple_eval_skipping > 0
            && std::abs(e.pos.simple_eval()) < config.simple_eval_skipping)
        {
            return true;
        }

        // Soft Early Ply Filter
        if (config.soft_early_fen_skipping > 0 && e.ply < config.soft_early_fen_skipping)
        {
            uint64_t ply_reject_threshold = static_cast<uint64_t>(
              (1.0 - early_ply_accept_prob[e.ply]) * static_cast<double>(~0ULL));
            if (prng() < ply_reject_threshold)
            {
                return true;
            }
        }

        // Dynamic Piece Count Filter
        const int pc = e.pos.piecesBB().count();
        if (pc < 0 || pc > 32)
            return true;

        static thread_local double   alpha                   = 1.0;
        static thread_local double   pc_history_all[33]      = {0};
        static thread_local double   pc_history_passed[33]   = {0};
        static thread_local double   pc_history_all_total    = 0;
        static thread_local double   pc_history_passed_total = 0;
        static thread_local uint64_t step_count              = 0;

        const double max_pc_skip_rate = 0.975;

        pc_history_all[pc] += 1.0;
        pc_history_all_total += 1.0;
        step_count++;

        bool should_update =
          (step_count == 100 || step_count == 500 || step_count == 1000 || step_count == 2500
           || step_count == 5000 || (step_count > 5000 && step_count % 10000 == 0));

        if (should_update)
        {
            double min_ratio   = std::numeric_limits<double>::infinity();
            bool   found_valid = false;

            for (int i = 0; i < 33; ++i)
            {
                if (target_pc_weights_lut[i] > 0.0 && pc_history_all[i] > 0.0)
                {
                    double current_ratio = (pc_history_all_total * target_pc_weights_lut[i])
                                         / (target_pc_weights_total * pc_history_all[i]);
                    if (current_ratio < min_ratio)
                    {
                        min_ratio   = current_ratio;
                        found_valid = true;
                    }
                }
            }
            if (found_valid && min_ratio > 0.0)
            {
                alpha = (1.0 - max_pc_skip_rate) / min_ratio;
            }

            if (step_count >= 10000 && step_count % 10000 == 0)
            {
                for (int i = 0; i < 33; ++i)
                {
                    pc_history_all[i] *= 0.5;
                }
                pc_history_all_total *= 0.5;
            }
        }

        double accept_prob = 0.0;
        if (target_pc_weights_lut[pc] > 0.0 && pc_history_all[pc] > 0.0)
        {
            double current_ratio = (pc_history_all_total * target_pc_weights_lut[pc])
                                 / (target_pc_weights_total * pc_history_all[pc]);
            accept_prob = alpha * current_ratio;
        }

        accept_prob = std::clamp(accept_prob, 0.0, 1.0);

        uint64_t reject_threshold =
          static_cast<uint64_t>((1.0 - accept_prob) * static_cast<double>(~0ULL));
        if (prng() < reject_threshold)
        {
            return true;
        }

        pc_history_passed[pc] += 1.0;
        pc_history_passed_total += 1.0;

        return false;
    };
}
