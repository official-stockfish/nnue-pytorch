import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from ..quantize import QuantizationManager


@contextmanager
def ieee_matmul_precision():
    """Forces standard IEEE float32 precision (disabling TF32) for numerical sensitivity.

    Ensures discrete routing decisions (argmin) and codebook distance metrics
    are bit-identical between training (CUDA), table generation (CUDA/CPU),
    and Stockfish inference.
    """
    if not torch.cuda.is_available():
        yield
        return

    prev_precision = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(prev_precision)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


NUM_MATERIAL_STATES = 378
NUM_OUR_KINGS = 33
NUM_ENEMY_KINGS = 65
ROUTER_TABLE_SIZE = NUM_MATERIAL_STATES * NUM_OUR_KINGS * NUM_ENEMY_KINGS  # 810,810
NUM_WEIGHT_EXPERTS = 64
NUM_BIAS_EXPERTS = 256
NUM_BIASES = 128
BIAS_RANK = 32
CODEBOOK_DIM = 256
ROUTER_HASH = 0x4475616C  # 'Dual'
DEFAULT_CODEBOOK_TARGET_NORM = 1.0


def rms_norm(
    x: torch.Tensor,
    target_norm: float = DEFAULT_CODEBOOK_TARGET_NORM,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Projects vectors onto a hypersphere of radius target_norm (parameter-free RMSNorm)."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    target_rms = target_norm / math.sqrt(x.shape[-1])
    return x * (target_rms / rms)


def center_and_rms_norm(
    x: torch.Tensor,
    target_norm: float = DEFAULT_CODEBOOK_TARGET_NORM,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean-centers vectors and projects them onto hypersphere of radius target_norm."""
    x_c = x - torch.mean(x, dim=-1, keepdim=True)
    rms = torch.sqrt(torch.mean(x_c**2, dim=-1, keepdim=True) + eps)
    target_rms = target_norm / math.sqrt(x.shape[-1])
    return x_c * (target_rms / rms)


def codebook_diversity_loss(cb: torch.Tensor) -> torch.Tensor:
    """Penalizes pairwise cosine similarity between distinct codebook vectors (Frobenius distance to identity)."""
    k = cb.shape[0]
    gram = torch.matmul(cb, cb.t())
    off_diag = gram - torch.eye(k, device=cb.device, dtype=cb.dtype)
    return torch.sum(off_diag**2) / (k * (k - 1))


def get_router_hash(
    num_weight_experts: int = NUM_WEIGHT_EXPERTS,
    num_bias_experts: int = NUM_BIAS_EXPERTS,
) -> int:
    """Computes the effective 32-bit router hash combining base 'Dual' and expert counts."""
    return ROUTER_HASH ^ ((num_bias_experts << 16) | num_weight_experts)


# Precompute bijective mapping for the 378 legal material states
_MATERIAL_STATES: list[tuple[int, int, int, int]] = []
for _t in range(16):
    for _p in range(min(8, _t) + 1):
        for _q in range(2):
            if _q == 1 and _t - _p < 1:
                continue
            for _bp in range(2):
                if _bp == 1 and (_t - _p - _q < 2):
                    continue
                _MATERIAL_STATES.append((_t, _p, _q, _bp))

assert len(_MATERIAL_STATES) == NUM_MATERIAL_STATES

_MATERIAL_ENCODE_TABLE = torch.full((16, 9, 2, 2), -1, dtype=torch.long)
for _idx, (_t, _p, _q, _bp) in enumerate(_MATERIAL_STATES):
    _MATERIAL_ENCODE_TABLE[_t, _p, _q, _bp] = _idx

_MATERIAL_DECODE_TABLE = torch.tensor(_MATERIAL_STATES, dtype=torch.long)


def encode_material_state(t: int, p: int, q: int, bp: int) -> int:
    """Encodes (T, P, Q, BP) into material ID in [0, 377]."""
    mat_id = int(_MATERIAL_ENCODE_TABLE[t, p, q, bp].item())
    if mat_id < 0:
        raise ValueError(f"Illegal material state: T={t}, P={p}, Q={q}, BP={bp}")
    return mat_id


def decode_material_state(mat_id: int) -> tuple[int, int, int, int]:
    """Decodes material ID in [0, 377] into (T, P, Q, BP)."""
    return tuple(_MATERIAL_STATES[mat_id])  # type: ignore[return-value]


def make_router_index(mat_id: int, our_king: int, enemy_king: int) -> int:
    """Computes the 1D context index in [0, 810809]."""
    return (mat_id * NUM_OUR_KINGS + our_king) * NUM_ENEMY_KINGS + enemy_king


def decompose_router_index(
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposes context index c into (mat_id, our_king, enemy_king)."""
    enemy_king = c % NUM_ENEMY_KINGS
    rem1 = torch.div(c, NUM_ENEMY_KINGS, rounding_mode="floor")
    our_king = rem1 % NUM_OUR_KINGS
    mat_id = torch.div(rem1, NUM_OUR_KINGS, rounding_mode="floor")
    return mat_id, our_king, enemy_king


class StructuredKingEmbedding(nn.Module):
    """
    Structured spatial embedding for our king and enemy king.
    Encodes:
      - Symmetric single-king geometry:
        - Our king: distance to home rank (0..7), distance to vertical edge (0..3), castling retention (0..1)
        - Enemy king: distance to enemy home rank (0..7), distance to vertical edge (0..3), castling retention (0..1)
      - 3-part relative interaction:
        - Vertical distance delta_r (0..7)
        - Horizontal distance delta_f (0..7)
        - Chebyshev opposition distance (2..7 mapped to 0..5, saving impossible 0 and 1)
    """

    def __init__(self, dim: int = CODEBOOK_DIM):
        super().__init__()
        self.dim = dim

        self.embed_our_rank = nn.Embedding(8, dim)
        self.embed_our_edge = nn.Embedding(4, dim)
        self.embed_our_castling = nn.Embedding(2, dim)

        self.embed_enemy_rank = nn.Embedding(8, dim)
        self.embed_enemy_edge = nn.Embedding(4, dim)
        self.embed_enemy_castling = nn.Embedding(2, dim)

        self.embed_delta_r = nn.Embedding(8, dim)
        self.embed_delta_f = nn.Embedding(8, dim)
        self.embed_chebyshev = nn.Embedding(6, dim)

        # Precompute lookup tables
        our_rank = torch.zeros(NUM_OUR_KINGS, dtype=torch.long)
        our_edge = torch.zeros(NUM_OUR_KINGS, dtype=torch.long)
        our_castling = torch.zeros(NUM_OUR_KINGS, dtype=torch.long)
        for k in range(NUM_OUR_KINGS):
            if k == 32:
                our_rank[k] = 0
                our_edge[k] = 3
                our_castling[k] = 1
            else:
                our_rank[k] = k // 4
                our_edge[k] = k % 4
                our_castling[k] = 0

        enemy_rank = torch.zeros(NUM_ENEMY_KINGS, dtype=torch.long)
        enemy_edge = torch.zeros(NUM_ENEMY_KINGS, dtype=torch.long)
        enemy_castling = torch.zeros(NUM_ENEMY_KINGS, dtype=torch.long)
        for k in range(NUM_ENEMY_KINGS):
            if k == 64:
                enemy_rank[k] = 0
                enemy_edge[k] = 3
                enemy_castling[k] = 1
            else:
                r = k // 8
                f = k % 8
                enemy_rank[k] = 7 - r
                enemy_edge[k] = min(f, 7 - f)
                enemy_castling[k] = 0

        delta_r = torch.zeros((NUM_OUR_KINGS, NUM_ENEMY_KINGS), dtype=torch.long)
        delta_f = torch.zeros((NUM_OUR_KINGS, NUM_ENEMY_KINGS), dtype=torch.long)
        chebyshev = torch.zeros((NUM_OUR_KINGS, NUM_ENEMY_KINGS), dtype=torch.long)
        for k_us in range(NUM_OUR_KINGS):
            r_us = 0 if k_us == 32 else k_us // 4
            f_us = 3 if k_us == 32 else k_us % 4
            for k_them in range(NUM_ENEMY_KINGS):
                r_them = 7 if k_them == 64 else k_them // 8
                f_them = 4 if k_them == 64 else k_them % 8
                dr = abs(r_us - r_them)
                df = min(abs(f_us - f_them), abs((7 - f_us) - f_them))
                d_cheb = max(dr, df)
                delta_r[k_us, k_them] = dr
                delta_f[k_us, k_them] = df
                # Legal chess distance between kings is >= 2. Clamping to min 2 and subtracting 2 yields index 0..5.
                chebyshev[k_us, k_them] = max(2, d_cheb) - 2

        self.register_buffer("our_rank", our_rank)
        self.register_buffer("our_edge", our_edge)
        self.register_buffer("our_castling", our_castling)
        self.register_buffer("enemy_rank", enemy_rank)
        self.register_buffer("enemy_edge", enemy_edge)
        self.register_buffer("enemy_castling", enemy_castling)
        self.register_buffer("delta_r", delta_r)
        self.register_buffer("delta_f", delta_f)
        self.register_buffer("chebyshev", chebyshev)
        self.init_weights()

    def init_weights(self) -> None:
        val = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.embed_our_rank.weight, -val, val)
        nn.init.uniform_(self.embed_our_edge.weight, -val, val)
        nn.init.uniform_(self.embed_our_castling.weight, -val, val)
        nn.init.uniform_(self.embed_enemy_rank.weight, -val, val)
        nn.init.uniform_(self.embed_enemy_edge.weight, -val, val)
        nn.init.uniform_(self.embed_enemy_castling.weight, -val, val)
        nn.init.uniform_(self.embed_delta_r.weight, -val, val)
        nn.init.uniform_(self.embed_delta_f.weight, -val, val)
        nn.init.uniform_(self.embed_chebyshev.weight, -val, val)

    def forward(self, k_us: torch.Tensor, k_them: torch.Tensor) -> torch.Tensor:
        z_our = (
            self.embed_our_rank(self.our_rank[k_us])
            + self.embed_our_edge(self.our_edge[k_us])
            + self.embed_our_castling(self.our_castling[k_us])
        )
        z_enemy = (
            self.embed_enemy_rank(self.enemy_rank[k_them])
            + self.embed_enemy_edge(self.enemy_edge[k_them])
            + self.embed_enemy_castling(self.enemy_castling[k_them])
        )
        z_rel = (
            self.embed_delta_r(self.delta_r[k_us, k_them])
            + self.embed_delta_f(self.delta_f[k_us, k_them])
            + self.embed_chebyshev(self.chebyshev[k_us, k_them])
        )
        return z_our + z_enemy + z_rel


class DualVQCodebookRouter(nn.Module):
    """
    Dual Discrete Vector Quantization (VQ) Codebook Router for Top-2 Downstream MoE:
    - Weight Router: K1 = 64 discrete experts (6 bits) selecting layer stack weights
    - Bias Router: K2 = 256 discrete experts (8 bits) selecting 128 dynamic threshold biases via rank-32 subspace
    - Single packed uint16 LUT of size 810,810 (1.55 MB)
    - 256 coalesced 128-element dynamic bias vectors (128 KB)
    """

    def __init__(
        self,
        num_contexts: int = ROUTER_TABLE_SIZE,
        num_weight_experts: int = NUM_WEIGHT_EXPERTS,
        num_bias_experts: int = NUM_BIAS_EXPERTS,
        dim: int = CODEBOOK_DIM,
        bias_rank: int = BIAS_RANK,
        num_biases: int = NUM_BIASES,
        commitment_cost: float = 0.005,
        balance_cost: float = 0.001,
        temperature: float = 0.2,
        diversity_cost: float = 0.01,
        reseed_dead_experts: bool = True,
        ema_decay: float = 0.999,
        quantization: QuantizationManager | None = None,
        codebook_target_norm: float = DEFAULT_CODEBOOK_TARGET_NORM,
    ):
        super().__init__()
        self.num_contexts = num_contexts
        self.num_weight_experts = num_weight_experts
        self.num_bias_experts = num_bias_experts
        self.dim = dim
        self.bias_rank = bias_rank
        self.num_biases = num_biases
        self.commitment_cost = commitment_cost
        self.balance_cost = balance_cost
        self.temperature = temperature
        self.diversity_cost = diversity_cost
        self.reseed_dead_experts = reseed_dead_experts
        self.ema_decay = ema_decay
        self.quantization = quantization
        self.codebook_target_norm = codebook_target_norm

        # Material feature embedding tables
        self.embed_T = nn.Embedding(16, dim)
        self.embed_P = nn.Embedding(9, dim)
        self.embed_Q = nn.Embedding(2, dim)
        self.embed_BP = nn.Embedding(2, dim)

        # Structured spatial King embedding table
        self.king_embed = StructuredKingEmbedding(dim=dim)

        # Weight Router Codebook: 64 experts in R^128
        self.codebook_weights = nn.Parameter(torch.empty(num_weight_experts, dim))

        # Bias Router Codebook: 512 experts in R^128
        self.codebook_biases = nn.Parameter(torch.empty(num_bias_experts, dim))
        self.bias_head_down = nn.Linear(dim, bias_rank, bias=False)
        self.bias_head_up = nn.Linear(bias_rank, num_biases, bias=True)

        # Material decoding buffer on device
        self.register_buffer(
            "material_decode_table", _MATERIAL_DECODE_TABLE, persistent=False
        )

        # Recent batch usage EMA buffers
        self.register_buffer(
            "ema_traffic_weights",
            torch.full(
                (num_weight_experts,), 1.0 / num_weight_experts, dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "ema_traffic_biases",
            torch.full(
                (num_bias_experts,), 1.0 / num_bias_experts, dtype=torch.float32
            ),
            persistent=False,
        )

        # Ephemeral runtime cache for inference & Stockfish export
        self.register_buffer(
            "lut",
            torch.zeros(num_contexts, dtype=torch.int32),
            persistent=False,
        )
        self.lut_valid: bool = False
        self.register_buffer(
            "coalesced_biases",
            torch.zeros(num_bias_experts, num_biases, dtype=torch.float32),
            persistent=False,
        )
        self.coalesced_valid: bool = False

        self.last_reseed_parents_w: dict[int, int] = {}
        self.last_reseed_parents_b: dict[int, int] = {}

        self.init_weights()

    @property
    def codebook_w(self) -> torch.Tensor:
        """Zero-mean RMS-normalized weight codebook (single source of truth)."""
        return center_and_rms_norm(self.codebook_weights, self.codebook_target_norm)

    @property
    def codebook_b(self) -> torch.Tensor:
        """Zero-mean RMS-normalized bias codebook (single source of truth)."""
        return center_and_rms_norm(self.codebook_biases, self.codebook_target_norm)

    @property
    def codebook(self) -> torch.Tensor:
        """Normalized weight codebook for downstream layer stacks and serialization."""
        return self.codebook_w

    def bias_head(self, emb: torch.Tensor) -> torch.Tensor:
        """Linear bottleneck on non-linearly activated codebook embeddings: Wup . Wdown . GELU(emb)."""
        h = self.bias_head_down(F.gelu(emb))
        h = rms_norm(h, target_norm=1.0)
        return self.bias_head_up(h)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.lut_valid = False
            self.coalesced_valid = False
        return self

    def init_weights(self) -> None:
        val = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.embed_T.weight, -val, val)
        nn.init.uniform_(self.embed_P.weight, -val, val)
        nn.init.uniform_(self.embed_Q.weight, -val, val)
        nn.init.uniform_(self.embed_BP.weight, -val, val)
        nn.init.uniform_(self.codebook_weights, -val, val)
        nn.init.uniform_(self.codebook_biases, -val, val)

        val_bias_down = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.bias_head_down.weight, -val_bias_down, val_bias_down)
        val_bias_up = 1.0 / math.sqrt(self.bias_rank)
        nn.init.uniform_(self.bias_head_up.weight, -val_bias_up, val_bias_up)
        nn.init.zeros_(self.bias_head_up.bias)

    def get_context_vector(self, c: torch.Tensor) -> torch.Tensor:
        """Computes, mean-centers, and RMS-normalizes context vector for batch of context indices c."""
        mat_id, k_us, k_them = decompose_router_index(c)
        mat_feats = self.material_decode_table[mat_id]  # (B, 4)
        t = mat_feats[:, 0]
        p = mat_feats[:, 1]
        q = mat_feats[:, 2]
        bp = mat_feats[:, 3]

        z_mat = self.embed_T(t) + self.embed_P(p) + self.embed_Q(q) + self.embed_BP(bp)
        z_king = self.king_embed(k_us, k_them)
        return center_and_rms_norm(z_mat + z_king, self.codebook_target_norm)

    def get_all_context_vectors(self) -> torch.Tensor:
        """Computes, mean-centers, and RMS-normalizes context vectors for all 810,810 contexts."""
        device = self.codebook_weights.device

        # Compute z_mat for all 378 material states
        mat_feats = self.material_decode_table  # (378, 4)
        z_mat = (
            self.embed_T(mat_feats[:, 0])
            + self.embed_P(mat_feats[:, 1])
            + self.embed_Q(mat_feats[:, 2])
            + self.embed_BP(mat_feats[:, 3])
        )  # (378, dim)

        # Compute z_king for all 33 x 65 = 2,145 king pairs
        all_kus = torch.arange(NUM_OUR_KINGS, device=device).repeat_interleave(
            NUM_ENEMY_KINGS
        )
        all_kth = torch.arange(NUM_ENEMY_KINGS, device=device).repeat(NUM_OUR_KINGS)
        z_king = self.king_embed(all_kus, all_kth)  # (2145, dim)

        # Outer sum: (378, 2145, dim) -> (810810, dim)
        z_all = (z_mat.unsqueeze(1) + z_king.unsqueeze(0)).reshape(
            self.num_contexts, self.dim
        )
        return center_and_rms_norm(z_all, self.codebook_target_norm)

    def quantize(
        self, z: torch.Tensor, codebook: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with ieee_matmul_precision():
            z_sq = torch.sum(z**2, dim=-1, keepdim=True)
            c_sq = torch.sum(codebook**2, dim=-1).unsqueeze(0)
            dot = torch.matmul(z, codebook.t())
            distances = z_sq + c_sq - 2.0 * dot

            indices = torch.argmin(distances, dim=-1)
            z_q = F.embedding(indices, codebook)

            # Commitment loss
            vq_loss = 0.25 * F.mse_loss(z, z_q.detach()) + F.mse_loss(z_q, z.detach())
            commit_loss = self.commitment_cost * vq_loss

            # Load balancing auxiliary loss
            k_exp = num_experts
            one_hot = F.one_hot(indices, num_classes=k_exp).float()
            f = one_hot.mean(dim=0).detach()

            scaled_neg_dist = -distances / max(self.temperature, 1e-4)
            probs = F.softmax(scaled_neg_dist, dim=-1)
            prob_mean = probs.mean(dim=0)

            balance_loss = k_exp * torch.sum(f * prob_mean) - 1.0

            # Straight-Through Estimator
            z_q_ste = z + (z_q - z).detach()

            return indices, z_q_ste, commit_loss, balance_loss

    def get_coalesced_bias_table(self) -> torch.Tensor:
        if self.coalesced_valid:
            return self.coalesced_biases
        with ieee_matmul_precision():
            raw_biases = self.bias_head(self.codebook_b)
            if self.quantization is not None:
                q_biases = self.quantization.quantize_dynamic_bias(raw_biases)
                self.coalesced_biases.copy_(
                    self.quantization.dequantize_dynamic_bias(q_biases)
                )
                self.coalesced_valid = True
                return self.coalesced_biases
            return raw_biases

    @torch.no_grad()
    def load_coalesced_biases(self, biases: torch.Tensor) -> None:
        self.coalesced_biases.copy_(
            biases.to(device=self.coalesced_biases.device, dtype=torch.float32)
        )
        self.coalesced_valid = True

    @torch.no_grad()
    def update_lookup_table(self) -> torch.Tensor:
        """Evaluates all 810,810 contexts and bit-packs (idx_w, idx_b) into uint16 lookup table."""
        with ieee_matmul_precision():
            z_all = self.get_all_context_vectors()

            cb_w = self.codebook_w
            cb_b = self.codebook_b
            cb_w_sq = torch.sum(cb_w**2, dim=-1).unsqueeze(0)
            cb_b_sq = torch.sum(cb_b**2, dim=-1).unsqueeze(0)

            chunk_size = 65536
            for i in range(0, self.num_contexts, chunk_size):
                z_chunk = z_all[i : i + chunk_size]
                z_sq = torch.sum(z_chunk**2, dim=-1, keepdim=True)

                # Weight routing (64 experts)
                dist_w = z_sq + cb_w_sq - 2.0 * torch.matmul(z_chunk, cb_w.t())
                idx_w = torch.argmin(dist_w, dim=-1)

                # Bias routing (256 experts)
                dist_b = z_sq + cb_b_sq - 2.0 * torch.matmul(z_chunk, cb_b.t())
                idx_b = torch.argmin(dist_b, dim=-1)

                # Bit packing: 6 bits weight | 8 bits bias
                packed = (idx_w & 0x3F) | ((idx_b & 0xFF) << 6)
                self.lut[i : i + chunk_size] = packed

            self.lut_valid = True

            if self.quantization is not None:
                raw_biases = self.bias_head(self.codebook_b)
                q_biases = self.quantization.quantize_dynamic_bias(raw_biases)
                self.coalesced_biases.copy_(
                    self.quantization.dequantize_dynamic_bias(q_biases)
                )
                self.coalesced_valid = True

            return self.lut

    def get_lookup_table(self) -> torch.Tensor:
        if not self.lut_valid:
            self.update_lookup_table()
        return self.lut

    @torch.no_grad()
    def load_lookup_table(self, table: torch.Tensor | bytes | bytearray) -> None:
        if isinstance(table, (bytes, bytearray)):
            table = torch.frombuffer(bytearray(table), dtype=torch.uint16).to(
                torch.int32
            )
        self.lut.copy_(table.to(device=self.lut.device, dtype=torch.int32))
        self.lut_valid = True
        for p in self.parameters():
            p.zero_()

    @torch.no_grad()
    def reseed_dead_experts_context_wide(
        self,
        temperature: float = 0.5,
        perturbation_scale: float = 0.02,
        min_utilization: float = 0.001,
    ) -> tuple[list[int], list[int]]:
        """Evaluates contexts across the board context space and reseeds dead experts
        via traffic-weighted cell splitting of active experts.
        """
        self.last_reseed_parents_w = {}
        self.last_reseed_parents_b = {}

        device = self.codebook_weights.device
        with ieee_matmul_precision():
            z_all = self.get_all_context_vectors()

            cb_w = self.codebook_w
            cb_b = self.codebook_b
            cb_w_sq = torch.sum(cb_w**2, dim=-1).unsqueeze(0)
            cb_b_sq = torch.sum(cb_b**2, dim=-1).unsqueeze(0)

            chunk_size = 65536
            all_idx_w = []
            all_idx_b = []

            for i in range(0, self.num_contexts, chunk_size):
                z_chunk = z_all[i : i + chunk_size]
                z_sq = torch.sum(z_chunk**2, dim=-1, keepdim=True)

                dist_w = z_sq + cb_w_sq - 2.0 * torch.matmul(z_chunk, cb_w.t())
                idx_w = torch.argmin(dist_w, dim=-1)
                all_idx_w.append(idx_w)

                dist_b = z_sq + cb_b_sq - 2.0 * torch.matmul(z_chunk, cb_b.t())
                idx_b = torch.argmin(dist_b, dim=-1)
                all_idx_b.append(idx_b)

            idx_w = torch.cat(all_idx_w, dim=0)
            idx_b = torch.cat(all_idx_b, dim=0)

        counts_w = torch.bincount(idx_w, minlength=self.num_weight_experts)
        dead_w = torch.nonzero(
            (counts_w == 0) | (self.ema_traffic_weights < min_utilization),
            as_tuple=True,
        )[0]
        reseeded_w: list[int] = []
        if len(dead_w) > 0:
            floor_w = (1.0 / self.num_weight_experts) - 1e-5
            active_w = torch.nonzero(
                self.ema_traffic_weights >= floor_w, as_tuple=True
            )[0]
            if len(active_w) == 0:
                active_w = torch.nonzero(counts_w > 0, as_tuple=True)[0]
            if len(active_w) == 0:
                active_w = torch.arange(self.num_weight_experts, device=device)
            if len(active_w) > 0:
                traffic_w = self.ema_traffic_weights[active_w].clone()
                if traffic_w.sum() <= 0:
                    traffic_w = counts_w[active_w].float()

                logits_w = torch.log(traffic_w.clamp(min=1e-8)) / max(temperature, 1e-4)
                probs_w = F.softmax(logits_w, dim=-1)

                modified_w = set(dead_w.tolist())
                parent_map_w: dict[int, int] = {}
                for dead_idx in dead_w:
                    target_pos = torch.multinomial(probs_w, num_samples=1).item()
                    target_idx = active_w[target_pos].item()
                    modified_w.add(target_idx)
                    parent_map_w[int(dead_idx)] = int(target_idx)

                    direction = torch.randn_like(self.codebook_weights[dead_idx])
                    norm = torch.norm(direction).clamp(min=1e-8)
                    eps = (direction / norm) * perturbation_scale

                    parent_vec = self.codebook_weights.data[target_idx].clone()
                    self.codebook_weights.data[dead_idx].copy_(parent_vec + eps)
                    self.codebook_weights.data[target_idx].copy_(parent_vec - eps)

                    # Split EMA traffic between parent and newly reseeded child
                    self.ema_traffic_weights[target_idx] *= 0.5
                    self.ema_traffic_weights[dead_idx] = self.ema_traffic_weights[
                        target_idx
                    ]

                    # Halve probability of the target to prevent dogpiling
                    probs_w[target_pos] *= 0.5
                    probs_w = probs_w / probs_w.sum().clamp(min=1e-8)

                reseeded_w = sorted(modified_w)
                self.last_reseed_parents_w = parent_map_w

        counts_b = torch.bincount(idx_b, minlength=self.num_bias_experts)
        dead_b = torch.nonzero(
            (counts_b == 0) | (self.ema_traffic_biases < min_utilization),
            as_tuple=True,
        )[0]
        reseeded_b: list[int] = []
        if len(dead_b) > 0:
            floor_b = (1.0 / self.num_bias_experts) - 1e-5
            active_b = torch.nonzero(self.ema_traffic_biases >= floor_b, as_tuple=True)[
                0
            ]
            if len(active_b) == 0:
                active_b = torch.nonzero(counts_b > 0, as_tuple=True)[0]
            if len(active_b) == 0:
                active_b = torch.arange(self.num_bias_experts, device=device)
            if len(active_b) > 0:
                traffic_b = self.ema_traffic_biases[active_b].clone()
                if traffic_b.sum() <= 0:
                    traffic_b = counts_b[active_b].float()

                logits_b = torch.log(traffic_b.clamp(min=1e-8)) / max(temperature, 1e-4)
                probs_b = F.softmax(logits_b, dim=-1)

                modified_b = set(dead_b.tolist())
                parent_map_b: dict[int, int] = {}
                for dead_idx in dead_b:
                    target_pos = torch.multinomial(probs_b, num_samples=1).item()
                    target_idx = active_b[target_pos].item()
                    modified_b.add(target_idx)
                    parent_map_b[int(dead_idx)] = int(target_idx)

                    direction = torch.randn_like(self.codebook_biases[dead_idx])
                    norm = torch.norm(direction).clamp(min=1e-8)
                    eps = (direction / norm) * perturbation_scale

                    parent_vec = self.codebook_biases.data[target_idx].clone()
                    self.codebook_biases.data[dead_idx].copy_(parent_vec + eps)
                    self.codebook_biases.data[target_idx].copy_(parent_vec - eps)

                    # Split EMA traffic between parent and newly reseeded child
                    self.ema_traffic_biases[target_idx] *= 0.5
                    self.ema_traffic_biases[dead_idx] = self.ema_traffic_biases[
                        target_idx
                    ]

                    # Halve probability of the target to prevent dogpiling
                    probs_b[target_pos] *= 0.5
                    probs_b = probs_b / probs_b.sum().clamp(min=1e-8)

                reseeded_b = sorted(modified_b)
                self.last_reseed_parents_b = parent_map_b

        if reseeded_w or reseeded_b:
            self.lut_valid = False
            self.coalesced_valid = False

        return reseeded_w, reseeded_b

    def forward(
        self,
        c_us: torch.Tensor,
        c_them: torch.Tensor,
        fake_quantize: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.training:
            c_combined = torch.cat([c_us, c_them], dim=0)
            z = self.get_context_vector(c_combined)

            e_w, _, commit_loss_w, balance_loss_w = self.quantize(
                z, self.codebook_w, self.num_weight_experts
            )
            idx_b, z_q_ste_b, commit_loss_b, balance_loss_b = self.quantize(
                z, self.codebook_b, self.num_bias_experts
            )

            # Update recent batch usage EMA buffers
            with torch.no_grad():
                freq_w = (
                    torch.bincount(e_w, minlength=self.num_weight_experts).float()
                    / e_w.numel()
                )
                freq_b = (
                    torch.bincount(idx_b, minlength=self.num_bias_experts).float()
                    / idx_b.numel()
                )
                self.ema_traffic_weights.lerp_(freq_w, 1.0 - self.ema_decay)
                self.ema_traffic_biases.lerp_(freq_b, 1.0 - self.ema_decay)

            dyn_bias = self.bias_head(z_q_ste_b)
            if fake_quantize and self.quantization is not None:
                dyn_bias = self.quantization.fake_quantize_dynamic_bias(dyn_bias)

            commit_loss = commit_loss_w + commit_loss_b
            balance_loss = balance_loss_w + balance_loss_b
            div_loss = codebook_diversity_loss(
                self.codebook_w
            ) + codebook_diversity_loss(self.codebook_b)
            moe_loss = (
                commit_loss
                + self.balance_cost * balance_loss
                + self.diversity_cost * div_loss
            )

            self.lut_valid = False
            self.coalesced_valid = False

            half = c_us.shape[0]
            e1_w = e_w[:half]
            e2_w = e_w[half:]
            dyn_bias_1 = dyn_bias[:half]
            dyn_bias_2 = dyn_bias[half:]

            b1_sqr = dyn_bias_1[:, 0:32]
            b1_lin = dyn_bias_1[:, 32:64]
            b2_sqr = dyn_bias_1[:, 64:96]
            b2_lin = dyn_bias_1[:, 96:128]
            dyn_biases_1 = (b1_sqr, b1_lin, b2_sqr, b2_lin)

            b1_sqr_2 = dyn_bias_2[:, 0:32]
            b1_lin_2 = dyn_bias_2[:, 32:64]
            b2_sqr_2 = dyn_bias_2[:, 64:96]
            b2_lin_2 = dyn_bias_2[:, 96:128]
            dyn_biases_2 = (b1_sqr_2, b1_lin_2, b2_sqr_2, b2_lin_2)

            return (
                e1_w,
                e2_w,
                dyn_biases_1,
                dyn_biases_2,
                moe_loss,
                commit_loss,
                balance_loss,
            )
        else:
            if not self.lut_valid:
                self.update_lookup_table()
            packed_1 = self.lut[c_us.long()]
            packed_2 = self.lut[c_them.long()]

            e1_w = (packed_1 & 0x3F).long()
            e2_w = (packed_2 & 0x3F).long()
            e1_b = ((packed_1 >> 6) & 0xFF).long()
            e2_b = ((packed_2 >> 6) & 0xFF).long()

            coalesced = self.get_coalesced_bias_table()
            dyn_bias_1 = coalesced[e1_b]
            dyn_bias_2 = coalesced[e2_b]

            dyn_biases_1 = (
                dyn_bias_1[:, 0:32],
                dyn_bias_1[:, 32:64],
                dyn_bias_1[:, 64:96],
                dyn_bias_1[:, 96:128],
            )
            dyn_biases_2 = (
                dyn_bias_2[:, 0:32],
                dyn_bias_2[:, 32:64],
                dyn_bias_2[:, 64:96],
                dyn_bias_2[:, 96:128],
            )

            zero = torch.tensor(0.0, device=c_us.device)
            return e1_w, e2_w, dyn_biases_1, dyn_biases_2, zero, zero, zero


# Backward-compatible alias for any modules importing VQCodebookRouter
VQCodebookRouter = DualVQCodebookRouter
