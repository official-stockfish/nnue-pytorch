import chess
import torch
from torch import nn


from ..feature_transformer import DoubleFeatureTransformer, SparseLinearFunction


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


def _orient(is_white_pov: bool, sq: int, ksq: int) -> int:
    kfile = ksq % 8
    return (7 * (kfile < 4)) ^ (56 * (not is_white_pov)) ^ sq


def _halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece) -> int:
    """Feature index using 12 piece types (no king merging)."""
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + KingBuckets[o_ksq] * 768


class HalfKav2Hm(DoubleFeatureTransformer):
    HASH = 0x7F234CB8
    FEATURE_NAME = "HalfKAv2_hm"
    MAX_ACTIVE_FEATURES = 32

    NUM_SQ = 64
    NUM_PT = 12
    NUM_PLANES = NUM_SQ * NUM_PT  # 768
    NUM_BUCKETS = NUM_SQ // 2  # 32
    NUM_INPUTS = NUM_PLANES * NUM_BUCKETS  # 24,576
    NUM_INPUTS_VIRTUAL = NUM_PLANES  # 768

    # Export size uses 11 piece types (704 * 32 = 22,528)
    NUM_REAL_FEATURES = 704 * 32  # 22,528

    def __init__(self, num_outputs: int):
        self.virtual_weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_outputs, dtype=torch.float32)
        )

        super().__init__(self.NUM_INPUTS, num_outputs)

    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        self.merged_weight = self.weight + self.virtual_weight.repeat(
            self.NUM_BUCKETS, 1
        )
        return (
            SparseLinearFunction.apply(
                feature_indices_0,
                feature_values_0,
                self.merged_weight,
                self.bias,
            ),
            SparseLinearFunction.apply(
                feature_indices_1,
                feature_values_1,
                self.merged_weight,
                self.bias,
            ),
        )

    @torch.no_grad()
    def coalesce(self) -> None:
        self.weight.add_(self.virtual_weight.repeat(self.NUM_BUCKETS, 1))
        self.virtual_weight.zero_()

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        """Initialize virtual weights to zero and set PSQT columns."""
        self.virtual_weight.zero_()

        scale = 1.0 / nnue2score
        L1 = self.num_outputs - num_psqt_buckets

        initial_values = self.halfka_psqts()
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
            self.bias[L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return coalesced weight remapped from 12→11 piece types for export.

        Returns (weight, psqt_weight) where weight has NUM_REAL_FEATURES rows
        and psqt_weight has NUM_REAL_FEATURES rows. Both are float tensors.
        The caller handles quantization.
        """
        # Coalesce virtual weights into a temporary copy
        coalesced = self.weight.data + self.virtual_weight.data.repeat(
            self.NUM_BUCKETS, 1
        )

        # Remap 12 piece types → 11 piece types
        export = coalesced.new_zeros(self.NUM_REAL_FEATURES, coalesced.shape[1])

        for b in range(self.NUM_BUCKETS):
            src_offset = b * self.NUM_PLANES  # 768 features per bucket
            dst_offset = b * 704  # 704 features per bucket in export

            # Copy first 10 piece types (p_idx 0..9) — 640 features
            export[dst_offset : dst_offset + 640] = coalesced[
                src_offset : src_offset + 640
            ]

            # Merge own king (p_idx=10) and opponent king (p_idx=11) into single block
            # p_idx 10 in 12pt: own king at all squares
            own_king_src = src_offset + 10 * 64
            opp_king_src = src_offset + 11 * 64
            dst_king = dst_offset + 10 * 64  # p_idx 10 in 11pt

            # For each square, the merged king block contains:
            # - own king at ksq (from p_idx 10, sq=ksq) — but ksq is implicit from bucket
            # - opponent king at all other squares (from p_idx 11)
            # Since both kings map to the same p_idx in 11pt,
            # we just add them (they never overlap for the same position)
            export[dst_king : dst_king + 64] = (
                coalesced[own_king_src : own_king_src + 64]
                + coalesced[opp_king_src : opp_king_src + 64]
            )

        return export

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        """Load export-format weights (11 piece types) and expand to 12.

        Takes a float tensor of shape (NUM_REAL_FEATURES, num_outputs).
        Expands 11→12 piece types and assigns to self.weight.
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
            # We can't perfectly reconstruct, so we put the merged weights
            # into both p_idx 10 and 11 (same as the old behavior)
            src_king = src_offset + 10 * 64
            expanded[dst_offset + 10 * 64 : dst_offset + 11 * 64] = export_weight[
                src_king : src_king + 64
            ]
            expanded[dst_offset + 11 * 64 : dst_offset + 12 * 64] = export_weight[
                src_king : src_king + 64
            ]

        self.weight.data.copy_(expanded)
        self.virtual_weight.zero_()

    def clip_weights(self) -> None:
        """No special clipping needed for HalfKav2Hm."""
        pass

    @staticmethod
    def halfka_psqts() -> list[int]:
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
                    idxw = _halfka_idx(True, ksq, s, chess.Piece(pt, chess.WHITE))
                    idxb = _halfka_idx(True, ksq, s, chess.Piece(pt, chess.BLACK))
                    values[idxw] = val
                    values[idxb] = -val

        return values
