import chess
import torch
from torch import nn


from ..feature_transformer import DoubleFeatureTransformer, SparseLinearFunction
from .halfka_v2_hm import _halfka_idx


class FullThreats(DoubleFeatureTransformer):
    HASH = 0x8F234CB8
    FEATURE_NAME = "Full_Threats"
    MAX_ACTIVE_FEATURES = 128 + 32

    NUM_SQ = 64
    NUM_PT = 12
    NUM_PLANES = NUM_SQ * NUM_PT  # 768
    NUM_BUCKETS = NUM_SQ // 2  # 32

    NUM_THREAT_FEATURES = 60_144
    NUM_PSQ_FEATURES = NUM_PLANES * NUM_BUCKETS  # 24,576
    NUM_INPUTS = NUM_THREAT_FEATURES + NUM_PSQ_FEATURES  # 84,720
    NUM_INPUTS_VIRTUAL = NUM_PLANES  # 768

    # Export size: threats + 11-piece-type PSQ (704 * 32 = 22,528)
    NUM_REAL_FEATURES = NUM_THREAT_FEATURES + 704 * 32  # 82,672

    def __init__(self, num_outputs: int):
        self.virtual_weight = nn.Parameter(
            torch.zeros(self.NUM_INPUTS_VIRTUAL, num_outputs, dtype=torch.float32)
        )

        super().__init__(self.NUM_INPUTS, num_outputs)

    def forward(
        self, feature_indices_0, feature_values_0, feature_indices_1, feature_values_1
    ):
        # Virtual weight applies only to the PSQ portion (repeated across buckets),
        # offset past the threat features
        psq_virtual = self.virtual_weight.repeat(self.NUM_BUCKETS, 1)
        threat_zeros = torch.zeros(
            self.NUM_THREAT_FEATURES,
            self.virtual_weight.shape[1],
            device=self.virtual_weight.device,
            dtype=self.virtual_weight.dtype,
        )
        full_virtual = torch.cat([threat_zeros, psq_virtual], dim=0)
        self.merged_weight = self.weight + full_virtual
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
        psq_virtual = self.virtual_weight.repeat(self.NUM_BUCKETS, 1)
        self.weight[self.NUM_THREAT_FEATURES :].add_(psq_virtual)
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
    def get_export_weights(self) -> torch.Tensor:
        """Return coalesced weight remapped from 12→11 piece types for export.

        Returns a float tensor with NUM_REAL_FEATURES rows.
        The threat portion is unchanged; the PSQ portion is remapped 12→11.
        """
        # Coalesce virtual weights into a temporary copy
        coalesced = self.weight.data.clone()
        psq_virtual = self.virtual_weight.data.repeat(self.NUM_BUCKETS, 1)
        coalesced[self.NUM_THREAT_FEATURES :].add_(psq_virtual)

        # Split into threat and PSQ portions
        threat_weight = coalesced[: self.NUM_THREAT_FEATURES]
        psq_weight = coalesced[self.NUM_THREAT_FEATURES :]

        # Remap PSQ portion 12→11 piece types
        psq_export = psq_weight.new_zeros(704 * 32, psq_weight.shape[1])

        for b in range(self.NUM_BUCKETS):
            src_offset = b * self.NUM_PLANES
            dst_offset = b * 704

            # Copy first 10 piece types
            psq_export[dst_offset : dst_offset + 640] = psq_weight[
                src_offset : src_offset + 640
            ]

            # Merge own king (p_idx=10) and opponent king (p_idx=11)
            own_king_src = src_offset + 10 * 64
            opp_king_src = src_offset + 11 * 64
            dst_king = dst_offset + 10 * 64

            psq_export[dst_king : dst_king + 64] = (
                psq_weight[own_king_src : own_king_src + 64]
                + psq_weight[opp_king_src : opp_king_src + 64]
            )

        return torch.cat([threat_weight, psq_export], dim=0)

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        """Load export-format weights (11 piece types for PSQ) and expand to 12.

        Takes a float tensor of shape (NUM_REAL_FEATURES, num_outputs).
        """
        threat_weight = export_weight[: self.NUM_THREAT_FEATURES]
        psq_export = export_weight[self.NUM_THREAT_FEATURES :]

        # Expand PSQ portion 11→12 piece types
        psq_expanded = psq_export.new_zeros(self.NUM_PSQ_FEATURES, psq_export.shape[1])

        for b in range(self.NUM_BUCKETS):
            src_offset = b * 704
            dst_offset = b * self.NUM_PLANES

            # Copy first 10 piece types
            psq_expanded[dst_offset : dst_offset + 640] = psq_export[
                src_offset : src_offset + 640
            ]

            # Split merged king block
            src_king = src_offset + 10 * 64
            psq_expanded[dst_offset + 10 * 64 : dst_offset + 11 * 64] = psq_export[
                src_king : src_king + 64
            ]
            psq_expanded[dst_offset + 11 * 64 : dst_offset + 12 * 64] = psq_export[
                src_king : src_king + 64
            ]

        expanded = torch.cat([threat_weight, psq_expanded], dim=0)
        self.weight.data.copy_(expanded)
        self.virtual_weight.zero_()

    def clip_weights(self) -> None:
        """Clamp threat weight slice to int8 range."""
        p = self.weight[: self.NUM_THREAT_FEATURES]
        p.data.clamp_(-128 / 255, 127 / 255)

    @staticmethod
    def halfka_psqts() -> list[int]:
        """PSQT initial values with threat feature prefix zeros.

        Returns NUM_INPUTS values (60,144 zeros + 24,576 PSQ values).
        """
        piece_values = {
            chess.PAWN: 126,
            chess.KNIGHT: 781,
            chess.BISHOP: 825,
            chess.ROOK: 1276,
            chess.QUEEN: 2538,
        }

        num_psq = 768 * 32  # 24,576
        num_total = 60_144 + num_psq
        values = [0] * num_total

        for ksq in range(64):
            for s in range(64):
                for pt, val in piece_values.items():
                    idxw = 60_144 + _halfka_idx(
                        True, ksq, s, chess.Piece(pt, chess.WHITE)
                    )
                    idxb = 60_144 + _halfka_idx(
                        True, ksq, s, chess.Piece(pt, chess.BLACK)
                    )
                    values[idxw] = val
                    values[idxb] = -val

        return values
