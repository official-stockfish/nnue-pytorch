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
    def init_weights(self) -> None:
        """Initialize virtual weights to zero."""
        self.zero_virtual_weights()

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
            ksq = InverseKingBuckets[b]

            export[dst_king : dst_king + 64] = coalesced[
                opp_king_src : opp_king_src + 64
            ]
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
            expanded[dst_offset : dst_offset + 640] = export_weight[
                src_offset : src_offset + 640
            ]

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

        self.weight.data.copy_(expanded)
        self.zero_virtual_weights()
