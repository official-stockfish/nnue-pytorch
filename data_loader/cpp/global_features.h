// Global NNUE input contract. Included after the upstream threat geometry tables.
// Copyright (C) 2026 The Stockfish developers. GPL-3.0-or-later.
#pragma once

template<bool Support>
struct GlobalRelationsExtractor final: IFeatureExtractor {
    static constexpr int RealRows = Support ? 14952 : 29904;
    int                  inputs() const override { return Support ? 2 * RealRows : RealRows; }
    int                  max_active_features() const override { return 256; }

    static int index(Color perspective, Piece a, Square from, Square to, Piece b) {
        bool friendly = a.color() == b.color();
        if (friendly != Support || a.type() == PieceType::King || b.type() == PieceType::King)
            return -1;
        const int owner = int(a.color());
        if constexpr (Support)
        {
            if (owner)
            {
                from = Square(int(from) ^ 56);
                to   = Square(int(to) ^ 56);
            }
        }
        else if (perspective == Color::Black)
        {
            from = Square(int(from) ^ 56);
            to   = Square(int(to) ^ 56);
            a    = Piece::fromId(int(a) ^ 1);
            b    = Piece::fromId(int(b) ^ 1);
        }
        if (a.type() == b.type() && from < to)
        {
            std::swap(a, b);
            std::swap(from, to);
        }
        int pt     = int(a.type());
        int mapped = FullThreats::map[pt][int(b.type())];
        if (mapped < 0)
            return -1;
        constexpr int Bases[5] = {0, 168, 1848, 4088, 7672};
        constexpr int Slots[5] = {84, 336, 560, 896, 1456};
        Bitboard      attacks =
          a.type() == PieceType::Pawn
            ? bb::pawnAttacks(Bitboard::square(from), Support ? Color::White : a.color())
            : bb::detail::pseudoAttacks()[a.type()][from];
        int row = Bases[pt] + mapped * Slots[pt] + threatoffsets[pt][int(from)]
                + (Bitboard::fromBits((1ULL << int(to)) - 1) & attacks).count();
        if constexpr (Support)
            return row + (owner != int(perspective) ? RealRows : 0);
        return row + int(a.color()) * 14952;
    }

    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color p) const override {
        int  n        = 0;
        auto occupied = e.pos.piecesBB() & ~e.pos.piecesBB(whiteKing) & ~e.pos.piecesBB(blackKing);
        for (int color = 0; color < 2; ++color)
            for (int pt = 0; pt < 5; ++pt)
            {
                Piece a{PieceType(pt), Color(color)};
                for (Square from : e.pos.piecesBB(a))
                    for (Square to : e.pos.attacks(from) & occupied)
                    {
                        Piece b = e.pos.pieceAt(to);
                        if (a.type() == b.type() && from < to)
                            continue;
                        int i = index(p, a, from, to, b);
                        if (i >= 0)
                            features[n++] = i;
                    }
            }
        return {n, inputs()};
    }
};

template<bool Frontier>
struct PawnStructureExtractor final: IFeatureExtractor {
    int        inputs() const override { return Frontier ? 16807 : 480; }
    int        max_active_features() const override { return Frontier ? 7 : 48; }
    static int rank_pair(int a, int b) {
        if (a > b)
            std::swap(a, b);
        return a * (11 - a) / 2 + b - a - 1;
    }
    static int pair_index(Color p, int f, int a, int b) {
        if (p == Color::Black)
        {
            a = (1 - a / 6) * 6 + 5 - a % 6;
            b = (1 - b / 6) * 6 + 5 - b % 6;
        }
        int i;
        if (a / 6 == b / 6)
            i = 15 * (a / 6) + rank_pair(a % 6, b % 6);
        else
        {
            if (a >= 6)
                std::swap(a, b);
            b -= 6;
            i = 30 + 5 * a + b - (b > a);
        }
        return 60 * f + i;
    }
    std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, Color p) const override {
        int states[8]{}, n = 0;
        for (int f = 0; f < 8; ++f)
        {
            int w = 0, b = 0, previous = -1;
            for (int r = 1; r <= 6; ++r)
            {
                Piece piece = e.pos.pieceAt(Square(8 * r + f));
                if (piece != whitePawn && piece != blackPawn)
                    continue;
                if (piece == whitePawn)
                    w = r;
                else if (!b)
                    b = r;
                if constexpr (!Frontier)
                {
                    int id = int(piece.color()) * 6 + r - 1;
                    if (previous >= 0)
                        features[n++] = pair_index(p, f, previous, id);
                    previous = id;
                }
            }
            auto reflect = [](int r) { return r ? 7 - r : 0; };
            states[f]    = p == Color::White ? 7 * w + b : 7 * reflect(b) + reflect(w);
        }
        if constexpr (Frontier)
            for (int f = 0; f < 7; ++f)
                features[n++] = 2401 * f + 49 * states[f] + states[f + 1];
        return {n, inputs()};
    }
};
