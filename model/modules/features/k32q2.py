import chess
import torch
from torch import nn

from .input_feature import InputFeature

# 32 Half-board King Buckets
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

# Relative direction vectors from king square to queen check square (up to 27 ray steps)
_RAY_DIRECTIONS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]

def _get_check_ray_index(ksq: int, check_sq: int) -> int:
    kf, kr = ksq % 8, ksq // 8
    cf, cr = check_sq % 8, check_sq // 8
    df, dr = cf - kf, cr - kr
    
    sf = 0 if df == 0 else (1 if df > 0 else -1)
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    
    dir_idx = -1
    for i, (dx, dy) in enumerate(_RAY_DIRECTIONS):
        if dx == sf and dy == sr:
            dir_idx = i
            break
            
    dist = max(abs(df), abs(dr)) - 1 # 0 to 6
    if dir_idx >= 0 and 0 <= dist < 7:
        idx = dir_idx * 3 + min(dist, 2)
        return min(idx, 26)
    return 0

def _k32q2_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece, opponent_has_queen: bool = False) -> int:
    """Feature index using 12 piece types (no king merging)."""
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    o_ksq = _orient(is_white_pov, king_sq, king_sq)
    k_bucket = KingBuckets[o_ksq]
    if k_bucket < 0:
        k_bucket = 0
    combined_bucket = k_bucket * 2 + (1 if opponent_has_queen else 0)
    return _orient(is_white_pov, sq, king_sq) + p_idx * 64 + combined_bucket * 768


class K32Q2(InputFeature):
    HASH = 0x32B5E284
    FEATURE_NAME = "K32Q2^"
    INPUT_FEATURE_NAME = "K32Q2"
    MAX_ACTIVE_FEATURES = 34
    EXPORT_WEIGHT_DTYPE = torch.int8

    NUM_SQ = 64
    NUM_PT = 12
    NUM_PLANES = NUM_SQ * NUM_PT  # 768
    NUM_BUCKETS = 64  # 32 KingBuckets * 2 QueenBuckets
    
    # 768 planes * 64 buckets = 49,152 piece-square virtual inputs
    NUM_PIECE_INPUTS = NUM_PLANES * NUM_BUCKETS  # 49,152
    NUM_QK4_INPUTS = 64 * 108 # 64 buckets * 27 rays * 4 states = 6,912
    
    NUM_INPUTS = NUM_PIECE_INPUTS + NUM_QK4_INPUTS  # 56,064
    NUM_INPUTS_VIRTUAL = NUM_PLANES  # 768

    # Export size uses 11 piece types (704 * 64 = 45,056) + 6,912 QK4 features = 51,968
    NUM_REAL_FEATURES = 704 * 64 + 6912  # 51,968

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
        v_expanded = torch.cat([
            self.virtual_weight.repeat(self.NUM_BUCKETS, 1),
            torch.zeros(self.NUM_QK4_INPUTS, self.num_outputs, device=self.weight.device, dtype=self.weight.dtype)
        ], dim=0)
        return self.weight + v_expanded

    @torch.no_grad()
    def coalesce(self) -> None:
        v_expanded = torch.cat([
            self.virtual_weight.repeat(self.NUM_BUCKETS, 1),
            torch.zeros(self.NUM_QK4_INPUTS, self.num_outputs, device=self.weight.device, dtype=self.weight.dtype)
        ], dim=0)
        self.weight.add_(v_expanded)
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

        initial_values = self.k32q2_psqts()
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
        Export weight size: 51,968.
        """
        coalesced = self.merged_weight()
        export = coalesced.new_zeros(self.NUM_REAL_FEATURES, coalesced.shape[1])

        # 1. Piece-Square Buckets (704 * 64 = 45,056)
        for b in range(self.NUM_BUCKETS):
            src_offset = b * self.NUM_PLANES
            dst_offset = b * 704

            # Copy first 10 piece types (640 features)
            export[dst_offset : dst_offset + 640] = coalesced[src_offset : src_offset + 640]

            # King block remapping
            own_king_src = src_offset + 10 * 64
            opp_king_src = src_offset + 11 * 64
            dst_king = dst_offset + 10 * 64

            export[dst_king : dst_king + 64] = coalesced[opp_king_src : opp_king_src + 64]

            k_bucket = b // 2
            for k in range(64):
                if KingBuckets[k] == k_bucket:
                    export[dst_king + k] = coalesced[own_king_src + k]

        # 2. QK4 Threat Features (6,912 features)
        qk4_src_offset = self.NUM_PIECE_INPUTS
        qk4_dst_offset = 704 * 64
        export[qk4_dst_offset:] = coalesced[qk4_src_offset:]

        return export

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        """Load export-format weights and expand to NUM_INPUTS (56,064)."""
        expanded = export_weight.new_zeros(self.NUM_INPUTS, export_weight.shape[1])

        # 1. Piece-Square Buckets
        for b in range(self.NUM_BUCKETS):
            src_offset = b * 704
            dst_offset = b * self.NUM_PLANES

            expanded[dst_offset : dst_offset + 640] = export_weight[src_offset : src_offset + 640]

            src_king = src_offset + 10 * 64
            k_bucket = b // 2

            for k in range(64):
                if KingBuckets[k] == k_bucket:
                    expanded[dst_offset + 10 * 64 + k] = export_weight[src_king + k]
                    expanded[dst_offset + 11 * 64 + k] = 0
                else:
                    expanded[dst_offset + 11 * 64 + k] = export_weight[src_king + k]

        # 2. QK4 Threat Features
        qk4_dst_offset = self.NUM_PIECE_INPUTS
        qk4_src_offset = 704 * 64
        expanded[qk4_dst_offset:] = export_weight[qk4_src_offset:]

        self.weight.data.copy_(expanded)
        self.zero_virtual_weights()

    def clip_weights(self, quantization) -> None:
        _i8 = torch.iinfo(torch.int8)
        min_w = -_i8.max / quantization.ft_quantized_one
        max_w = _i8.max / quantization.ft_quantized_one
        self.weight.data.clamp_(min_w, max_w)

    @staticmethod
    def k32q2_psqts() -> list[int]:
        """PSQT initial values for K32Q2 (56,064 values)."""
        piece_values = {
            chess.PAWN: 126,
            chess.KNIGHT: 781,
            chess.BISHOP: 825,
            chess.ROOK: 1276,
            chess.QUEEN: 2538,
        }

        num_inputs = 56064
        values = [0] * num_inputs

        for ksq in range(64):
            for s in range(64):
                for pt, val in piece_values.items():
                    for q in [False, True]:
                        idxw = _k32q2_idx(True, ksq, s, chess.Piece(pt, chess.WHITE), q)
                        idxb = _k32q2_idx(True, ksq, s, chess.Piece(pt, chess.BLACK), q)
                        values[idxw] = val
                        values[idxb] = -val

        return values
