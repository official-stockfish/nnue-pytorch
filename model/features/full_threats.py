from collections import OrderedDict

import chess

from .feature_block import FeatureBlock

SQUARE_NB = 64
PIECE_NB = 12
COLOR_NB = 2
PIECE_TYPE_NB = 8
MAX_ACTIVE_FEATURES = 128+32
"""
OrientTBL = [
[ 
    a1, a1, a1, a1, h1, h1, h1, h1,
    a1, a1, a1, a1, h1, h1, h1, h1,
    a1, a1, a1, a1, h1, h1, h1, h1,
    a1, a1, a1, a1, h1, h1, h1, h1,
    a1, a1, a1, a1, h1, h1, h1, h1,
    a1, a1, a1, a1, h1, h1, h1, h1,
    a1, a1, a1, a1, h1, h1, h1, h1,
    a1, a1, a1, a1, h1, h1, h1, h1 ],
[ 
    a8, a8, a8, a8, h8, h8, h8, h8,
    a8, a8, a8, a8, h8, h8, h8, h8,
    a8, a8, a8, a8, h8, h8, h8, h8,
    a8, a8, a8, a8, h8, h8, h8, h8,
    a8, a8, a8, a8, h8, h8, h8, h8,
    a8, a8, a8, a8, h8, h8, h8, h8,
    a8, a8, a8, a8, h8, h8, h8, h8,
    a8, a8, a8, a8, h8, h8, h8, h8 ]
]

    
int threatoffsets[PIECE_NB][SQUARE_NB+2];
void init_threat_offsets() {
    int pieceoffset = 0;
    Piece piecetbl[12] = {whitePawn, blackPawn, whiteKnight, blackKnight, whiteBishop,
    blackBishop, whiteRook, blackRook, whiteQueen, blackQueen, whiteKing, blackKing};
    for (int piece = 0; piece < 12; piece++) {
        threatoffsets[piece][65] = pieceoffset;
        int squareoffset = 0;
        for (int from = (int)a1; from <= (int)h8; from++) {
            threatoffsets[piece][from] = squareoffset;
            if (piecetbl[piece].type() != PieceType::Pawn) {
                Bitboard attacks = bb::detail::pseudoAttacks()[piecetbl[piece].type()][Square(from)];
                squareoffset += attacks.count();
            }
            else if (from >= (int)a2 && from <= (int)h7) {
                Bitboard attacks = bb::pawnAttacks(Bitboard::square(Square(from)), piecetbl[piece].color());
                squareoffset += attacks.count();
            }
        }
        threatoffsets[piece][64] = squareoffset;
        pieceoffset += numvalidtargets[piece]*squareoffset;
    }
}
"""
numvalidtargets = [6, 6, 12, 12, 10, 10, 10, 10, 12, 12, 8, 8]
map = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, -1, 4],
    [0, 1, 2, 3, -1, 4],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, -1, -1]
]

NUM_SQ = 64
NUM_PT_REAL = 11
NUM_PT_VIRTUAL = 12
NUM_PLANES_REAL = NUM_SQ * NUM_PT_REAL
NUM_PLANES_VIRTUAL = NUM_SQ * NUM_PT_VIRTUAL
NUM_INPUTS = 79856 + NUM_PLANES_REAL * NUM_SQ // 2

# fmt: off
KingBuckets = [
  -1, -1, -1, -1, 31, 30, 29, 28,
  -1, -1, -1, -1, 27, 26, 25, 24,
  -1, -1, -1, -1, 23, 22, 21, 20,
  -1, -1, -1, -1, 19, 18, 17, 16,
  -1, -1, -1, -1, 15, 14, 13, 12,
  -1, -1, -1, -1, 11, 10, 9, 8,
  -1, -1, -1, -1, 7, 6, 5, 4,
  -1, -1, -1, -1, 3, 2, 1, 0
]
# fmt: on


def orient(is_white_pov: bool, sq: int, ksq: int) -> int:
    # ksq must not be oriented
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece) -> int:
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    o_ksq = orient(is_white_pov, king_sq, king_sq)
    if p_idx == 11:
        p_idx -= 1
    return (
        79856
        + orient(is_white_pov, sq, king_sq)
        + p_idx * NUM_SQ
        + KingBuckets[o_ksq] * NUM_PLANES_REAL
    )


def halfka_psqts() -> list[int]:
    # values copied from stockfish, in stockfish internal units
    piece_values = {
        chess.PAWN: 126,
        chess.KNIGHT: 781,
        chess.BISHOP: 825,
        chess.ROOK: 1276,
        chess.QUEEN: 2538,
    }

    values = [0] * NUM_INPUTS

    for ksq in range(64):
        for s in range(64):
            for pt, val in piece_values.items():
                idxw = halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
                idxb = halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
                values[idxw] = val
                values[idxb] = -val

    return values


class Features(FeatureBlock):
    def __init__(self):
        super().__init__(
            "Full_Threats", 0x8F234CB8, OrderedDict([("Full_Threats", NUM_INPUTS)])
        )

    def get_active_features(self, board: chess.Board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for support during training"
        )

    def get_initial_psqt_features(self) -> list[int]:
        return halfka_psqts()


class FactorizedFeatures(FeatureBlock):
    def __init__(self):
        super().__init__(
            "Full_Threats^",
            0x8F234CB8,
            OrderedDict([("Full_Threats", NUM_INPUTS), ("A", NUM_PLANES_VIRTUAL)]),
        )

    def get_active_features(self, board: chess.Board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for factorizer support during training"
        )

    def get_feature_factors(self, idx: int) -> list[int]:
        if idx >= self.num_real_features:
            raise Exception("Feature must be real")
        if idx < 79856:
            return [idx]
        
        a_idx = (idx - 79856) % NUM_PLANES_REAL
        k_idx = (idx - 79856) // NUM_PLANES_REAL

        if a_idx // NUM_SQ == 10 and k_idx != KingBuckets[a_idx % NUM_SQ]:
            a_idx += NUM_SQ

        return [idx, self.get_factor_base_feature("A") + a_idx]

    def get_initial_psqt_features(self) -> list[int]:
        return halfka_psqts() + [0] * NUM_PLANES_VIRTUAL


"""
This is used by the features module for discovery of feature blocks.
"""


def get_feature_block_clss() -> list[type[FeatureBlock]]:
    return [Features, FactorizedFeatures]
