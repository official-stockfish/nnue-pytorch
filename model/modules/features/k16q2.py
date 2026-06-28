import chess
import torch
from torch import nn

from .input_feature import InputFeature


# fmt: off
KingBuckets = [
  -1, -1, -1, -1, 14, 14, 15, 15,
  -1, -1, -1, -1, 14, 14, 15, 15,
  -1, -1, -1, -1, 12, 12, 13, 13,
  -1, -1, -1, -1, 12, 12, 13, 13,
  -1, -1, -1, -1,  8,  9, 10, 11,
  -1, -1, -1, -1,  8,  9, 10, 11,
  -1, -1, -1, -1,  4,  5,  6,  7,
  -1, -1, -1, -1,  0,  1,  2,  3
]
# fmt: on

# Inverse mapping: king_bucket -> oriented king square
InverseKingBuckets = [0] * 16
for _sq, _bucket in enumerate(KingBuckets):
    if _bucket >= 0:
        InverseKingBuckets[_bucket] = _sq


def _orient(is_white_pov: bool, sq: int, ksq: int) -> int:
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def _k16q2_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece) -> int:
    """Feature index using 12 piece types (no king merging)."""
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    k_bucket = KingBuckets[o_ksq]
    # PSQT init values are computed for empty boards, so no opponent queen.
    opponent_has_queen = False
    combined_bucket = k_bucket * 2 + (1 if opponent_has_queen else 0)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + combined_bucket * 768


class K16Q2(InputFeature):
    HASH = 0x32B5E284
    FEATURE_NAME = "K16Q2^"
    INPUT_FEATURE_NAME = "K16Q2"
    MAX_ACTIVE_FEATURES = 32

    NUM_SQ = 64
    NUM_PT = 12
    NUM_PLANES = NUM_SQ * NUM_PT  # 768
    NUM_BUCKETS = 32  # 16 KingBuckets * 2 QueenBuckets
    NUM_INPUTS = NUM_PLANES * NUM_BUCKETS  # 24,576
    NUM_INPUTS_VIRTUAL = NUM_PLANES  # 768

    # Export size uses 11 piece types (704 * 32 = 22,528)
    NUM_REAL_FEATURES = 704 * 32  # 22,528

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )
        self.virtual_weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.virtual_weight.repeat(self.NUM_BUCKETS, 1)

    @torch.no_grad()
    def coalesce(self) -> None:
        self.weight.add_(self.virtual_weight.repeat(self.NUM_BUCKETS, 1))
        self.zero_virtual_weights()

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.virtual_weight.zero_()

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        """Initialize virtual weights to zero and set PSQT columns."""
        self.zero_virtual_weights()

        scale = 1.0 / nnue2score
        L1 = self.num_outputs - num_psqt_buckets

        initial_values = self.k16q2_psqts()
        assert len(initial_values) == self.NUM_INPUTS

        new_weights = (
            torch.tensor(
                initial_values,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            * scale
        )

        for i in range(num_psqt_buckets):
            self.weight[:, L1 + i] = new_weights

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        """Return coalesced weight remapped from 12->11 piece types for export.

        Returns a float tensor with NUM_REAL_FEATURES rows.
        """
        # Coalesce virtual weights
        coalesced = self.merged_weight()

        # Remap 12 piece types -> 11 piece types
        export = coalesced.new_zeros(self.NUM_REAL_FEATURES, coalesced.shape[1])

        for b in range(self.NUM_BUCKETS):
            src_offset = b * self.NUM_PLANES  # 768 features per bucket
            dst_offset = b * 704  # 704 features per bucket in export

            # Copy first 10 piece types (p_idx 0..9) -- 640 features
            export[dst_offset : dst_offset + 640] = coalesced[
                src_offset : src_offset + 640
            ]

            # Merge own king (p_idx=10) and opponent king (p_idx=11) into single block
            own_king_src = src_offset + 10 * 64
            opp_king_src = src_offset + 11 * 64
            dst_king = dst_offset + 10 * 64

            export[dst_king : dst_king + 64] = coalesced[
                opp_king_src : opp_king_src + 64
            ]

            # Copy own king weights for all squares in this bucket
            k_bucket = b // 2
            for k in range(64):
                if KingBuckets[k] == k_bucket:
                    export[dst_king + k] = coalesced[own_king_src + k]

        return export

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        """Load export-format weights (11 piece types) and expand to 12.

        Takes a float tensor of shape (NUM_REAL_FEATURES, num_outputs).
        Expands 11->12 piece types and assigns to self.weight.
        Zeros self.virtual_weight.
        """
        expanded = export_weight.new_zeros(self.NUM_INPUTS, export_weight.shape[1])

        for b in range(self.NUM_BUCKETS):
            src_offset = b * 704
            dst_offset = b * self.NUM_PLANES

            # Copy first 10 piece types
            expanded[dst_offset : dst_offset + 640] = export_weight[
                src_offset : src_offset + 640
            ]

            # Split merged king block back into p_idx 10 and 11
            src_king = src_offset + 10 * 64
            k_bucket = b // 2

            for k in range(64):
                if KingBuckets[k] == k_bucket:
                    # Own king: only weight at k matters (rest stays zero)
                    expanded[dst_offset + 10 * 64 + k] = export_weight[src_king + k]
                    # Opponent king: 0 for these squares
                    expanded[dst_offset + 11 * 64 + k] = 0
                else:
                    # Opponent king: copy from merged
                    expanded[dst_offset + 11 * 64 + k] = export_weight[src_king + k]

        self.weight.data.copy_(expanded)
        self.zero_virtual_weights()

    @staticmethod
    def k16q2_psqts() -> list[int]:
        """PSQT initial values using 12 piece types (24,576 values)."""
        piece_values = {
            chess.PAWN: 126,
            chess.KNIGHT: 781,
            chess.BISHOP: 825,
            chess.ROOK: 1276,
            chess.QUEEN: 2538,
        }

        num_inputs = 768 * 32  # 24,576
        values = [0] * num_inputs

        for ksq in range(64):
            for s in range(64):
                for pt, val in piece_values.items():
                    idxw = _k16q2_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
                    idxb = _k16q2_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
                    values[idxw] = val
                    values[idxb] = -val

        return values
