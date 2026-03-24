import math
import chess
import torch
from torch import nn

from .input_feature import InputFeature


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

# Inverse mapping: bucket -> oriented king square
InverseKingBuckets = [0] * 32
for _sq, _bucket in enumerate(KingBuckets):
    if _bucket >= 0:
        InverseKingBuckets[_bucket] = _sq


def _orient(is_white_pov: bool, sq: int, ksq: int) -> int:
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def _halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece) -> int:
    """Feature index using 12 piece types (no king merging)."""
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KingBuckets[o_ksq] * 768


class HalfKav2Hm(InputFeature):
    HASH = 0x7F234CB8
    FEATURE_NAME = "HalfKAv2_hm^"
    INPUT_FEATURE_NAME = "HalfKAv2_hm"
    MAX_ACTIVE_FEATURES = 32

    NUM_SQ = 64
    NUM_PT = 12
    NUM_PLANES = NUM_SQ * NUM_PT  # 768
    NUM_BUCKETS = NUM_SQ // 2  # 32
    NUM_INPUTS = NUM_PLANES * NUM_BUCKETS  # 24,576
    NUM_INPUTS_VIRTUAL = NUM_PLANES  # 768

    # Export size uses 11 piece types (704 * 32 = 22,528)
    NUM_REAL_FEATURES = 704 * 32  # 22,528

    def __init__(self, l1: int, num_psqt_buckets: int):
        super().__init__()

        self.l1 = l1
        self.num_psqt_buckets = num_psqt_buckets
        self.num_outputs = l1 + num_psqt_buckets

        self.weight_ft = nn.Parameter(
            torch.empty(self.NUM_INPUTS, l1, dtype=torch.float32)
        )
        self.virtual_weight_ft = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, l1, dtype=torch.float32)
        )

        self.weight_psqt = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_psqt_buckets, dtype=torch.float32)
        )
        self.virtual_weight_psqt = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_psqt_buckets, dtype=torch.float32)
        )

        self.reset_parameters()

    def get_ft_params(self) -> list[nn.Parameter]:
        return [self.weight_ft, self.virtual_weight_ft]

    def get_psqt_params(self) -> list[nn.Parameter]:
        return [self.weight_psqt, self.virtual_weight_psqt]

    @torch.no_grad()
    def reset_parameters(self) -> None:
        sigma = math.sqrt(1 / self.NUM_INPUTS)
        self.weight_ft.uniform_(-sigma, sigma)
        self.weight_psqt.uniform_(-sigma, sigma)
        self.virtual_weight_ft.zero_()
        self.virtual_weight_psqt.zero_()

    def merged_weight(self) -> torch.Tensor:
        merged_ft = self.weight_ft + self.virtual_weight_ft.repeat(self.NUM_BUCKETS, 1)
        merged_psqt = self.weight_psqt + self.virtual_weight_psqt.repeat(self.NUM_BUCKETS, 1)
        return torch.cat([merged_ft, merged_psqt], dim=1)

    @torch.no_grad()
    def coalesce(self) -> None:
        self.weight_ft.add_(self.virtual_weight_ft.repeat(self.NUM_BUCKETS, 1))
        self.virtual_weight_ft.zero_()
        self.weight_psqt.add_(self.virtual_weight_psqt.repeat(self.NUM_BUCKETS, 1))
        self.virtual_weight_psqt.zero_()

    @torch.no_grad()
    def init_weights(self, nnue2score: float) -> None:
        """Initialize virtual weights to zero and set PSQT columns."""
        self.virtual_weight_ft.zero_()
        self.virtual_weight_psqt.zero_()

        scale = 1.0 / nnue2score
        initial_values = self.halfka_psqts()
        assert len(initial_values) == self.NUM_INPUTS

        new_weights = (
            torch.tensor(
                initial_values,
                device=self.weight_psqt.device,
                dtype=self.weight_psqt.dtype,
            )
            * scale
        )

        for i in range(self.num_psqt_buckets):
            self.weight_psqt[:, i] = new_weights

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        """Return coalesced weight remapped from 12->11 piece types for export.

        Returns a float tensor with NUM_REAL_FEATURES rows.
        """
        coalesced_ft = self.weight_ft.data + self.virtual_weight_ft.data.repeat(self.NUM_BUCKETS, 1)
        coalesced_psqt = self.weight_psqt.data + self.virtual_weight_psqt.data.repeat(self.NUM_BUCKETS, 1)
        coalesced = torch.cat([coalesced_ft, coalesced_psqt], dim=1)

        export = coalesced.new_zeros(self.NUM_REAL_FEATURES, coalesced.shape[1])

        for b in range(self.NUM_BUCKETS):
            src_offset = b * self.NUM_PLANES
            dst_offset = b * 704

            export[dst_offset : dst_offset + 640] = coalesced[src_offset : src_offset + 640]

            own_king_src = src_offset + 10 * 64
            opp_king_src = src_offset + 11 * 64
            dst_king = dst_offset + 10 * 64
            ksq = InverseKingBuckets[b]

            export[dst_king : dst_king + 64] = coalesced[opp_king_src : opp_king_src + 64]
            export[dst_king + ksq] = coalesced[own_king_src + ksq]

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
            expanded[dst_offset : dst_offset + 640] = export_weight[src_offset : src_offset + 640]

            # Split merged king block back into p_idx 10 and 11
            src_king = src_offset + 10 * 64
            ksq = InverseKingBuckets[b]

            # Own king: only weight at ksq matters (rest stays zero)
            expanded[dst_offset + 10 * 64 + ksq] = export_weight[src_king + ksq]

            # Opponent king: all squares from merged, except ksq -> 0
            expanded[dst_offset + 11 * 64 : dst_offset + 12 * 64] = export_weight[
                src_king : src_king + 64
            ]
            expanded[dst_offset + 11 * 64 + ksq] = 0

        self.weight_ft.data.copy_(expanded[:, :self.l1])
        self.weight_psqt.data.copy_(expanded[:, self.l1:])
        self.virtual_weight_ft.zero_()
        self.virtual_weight_psqt.zero_()

    @staticmethod
    def halfka_psqts() -> list[int]:
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
                    idxw = _halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
                    idxb = _halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
                    values[idxw] = val
                    values[idxb] = -val

        return values
