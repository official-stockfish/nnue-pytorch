import chess
from collections import OrderedDict
from feature_block import *

NUM_SQ = 64
NUM_PT_REAL = 11
NUM_PT_VIRTUAL = 12
NUM_PLANES_REAL = NUM_SQ * NUM_PT_REAL
NUM_PLANES_VIRTUAL = NUM_SQ * NUM_PT_VIRTUAL
NUM_INPUTS = NUM_PLANES_REAL * NUM_SQ // 2

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


def orient(is_white_pov: bool, sq: int, ksq: int):
    # ksq must not be oriented
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    o_ksq = orient(is_white_pov, king_sq, king_sq)
    if p_idx == 11:
        p_idx -= 1
    return (
        orient(is_white_pov, sq, king_sq)
        + p_idx * NUM_SQ
        + KingBuckets[o_ksq] * NUM_PLANES_REAL
    )


def halfka_psqts():
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
        super(Features, self).__init__(
            "HalfKAv2_hm", 0x7F234CB8, OrderedDict([("HalfKAv2_hm", NUM_INPUTS)])
        )

    def get_active_features(self, board: chess.Board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for support during training"
        )

    def get_initial_psqt_features(self):
        return halfka_psqts()


class FactorizedFeatures(FeatureBlock):
    def __init__(self):
        super(FactorizedFeatures, self).__init__(
            "HalfKAv2_hm^",
            0x7F234CB8,
            OrderedDict([("HalfKAv2_hm", NUM_INPUTS), ("A", NUM_PLANES_VIRTUAL)]),
        )

    def get_active_features(self, board: chess.Board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for factorizer support during training"
        )

    def get_feature_factors(self, idx):
        if idx >= self.num_real_features:
            raise Exception("Feature must be real")

        a_idx = idx % NUM_PLANES_REAL
        k_idx = idx // NUM_PLANES_REAL

        if a_idx // NUM_SQ == 10 and k_idx != KingBuckets[a_idx % NUM_SQ]:
            a_idx += NUM_SQ

        return [idx, self.get_factor_base_feature("A") + a_idx]

    def get_initial_psqt_features(self):
        return halfka_psqts() + [0] * NUM_PLANES_VIRTUAL


"""
This is used by the features module for discovery of feature blocks.
"""


def get_feature_block_clss():
    return [Features, FactorizedFeatures]
