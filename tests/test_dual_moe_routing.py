import io

import pytest
import torch

from cross_check_eval import init_random_weights
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.router import (
    NUM_BIAS_EXPERTS,
    NUM_BIASES,
    NUM_MATERIAL_STATES,
    NUM_WEIGHT_EXPERTS,
    ROUTER_HASH,
    ROUTER_TABLE_SIZE,
    DualVQCodebookRouter,
    StructuredKingEmbedding,
    decode_material_state,
    encode_material_state,
    get_router_hash,
)
from model.utils.serialize import NNUEReader, NNUEWriter


def test_material_bijection_exhaustive():
    """Verify that all 378 valid legal material states map bijectively to [0, 377]."""
    seen_ids = set()
    for m in range(NUM_MATERIAL_STATES):
        t, p, q, bp = decode_material_state(m)
        assert 0 <= t <= 15
        assert 0 <= p <= min(8, t)
        if q == 1:
            assert t - p >= 1
        if bp == 1:
            assert t - p - q >= 2

        encoded_id = encode_material_state(t, p, q, bp)
        assert encoded_id == m
        seen_ids.add(encoded_id)

    assert len(seen_ids) == 378

    # Verify illegal states raise ValueError
    with pytest.raises(ValueError):
        encode_material_state(0, 0, 1, 0)  # Q=1 but T-P=0
    with pytest.raises(ValueError):
        encode_material_state(1, 0, 0, 1)  # BP=1 but T-P-Q=1


def test_structured_king_geometry():
    """Verify StructuredKingEmbedding dimensions, lookups, and Chebyshev opposition."""
    emb = StructuredKingEmbedding(dim=256)

    # Test our king bucket 32 (e1 with castling)
    assert emb.our_rank[32].item() == 0
    assert emb.our_edge[32].item() == 3
    assert emb.our_castling[32].item() == 1

    # Test enemy king bucket 64 (e8 with castling)
    assert emb.enemy_rank[64].item() == 0  # distance to enemy home rank (rank 8)
    assert emb.enemy_edge[64].item() == 3
    assert emb.enemy_castling[64].item() == 1

    # Test delta_r table bounds
    assert emb.delta_r.min().item() == 0
    assert emb.delta_r.max().item() == 7
    assert emb.delta_r.shape == (33, 65)

    # Test delta_f table bounds
    assert emb.delta_f.min().item() == 0
    assert emb.delta_f.max().item() <= 7
    assert emb.delta_f.shape == (33, 65)

    # Test Chebyshev table bounds (mapped 2..7 -> 0..5)
    assert emb.chebyshev.min().item() == 0
    assert emb.chebyshev.max().item() == 5
    assert emb.chebyshev.shape == (33, 65)

    # Test forward
    kus = torch.tensor([0, 15, 32])
    kth = torch.tensor([0, 31, 64])
    z_k = emb(kus, kth)
    assert z_k.shape == (3, 256)


def test_dual_router_training_gradient_flow():
    """Verify forward and backward gradient flow into codebooks and embeddings."""
    torch.manual_seed(123)
    router = DualVQCodebookRouter()
    router.train()

    c_us = torch.tensor([0, 1000, 50000, 800000], dtype=torch.long)
    c_them = torch.tensor([800000, 50000, 1000, 0], dtype=torch.long)

    (
        e1_w,
        e2_w,
        dyn_biases_1,
        dyn_biases_2,
        moe_loss,
        commit_loss,
        balance_loss,
    ) = router(c_us, c_them)

    assert e1_w.shape == (4,)
    assert e2_w.shape == (4,)
    assert all(0 <= idx.item() < NUM_WEIGHT_EXPERTS for idx in e1_w)
    assert all(0 <= idx.item() < NUM_WEIGHT_EXPERTS for idx in e2_w)

    for b in dyn_biases_1 + dyn_biases_2:
        assert b.shape == (4, 32)

    assert moe_loss.ndim == 0
    assert commit_loss.ndim == 0
    assert balance_loss.ndim == 0

    # Backprop
    total_loss = (
        moe_loss
        + dyn_biases_1[0].sum()
        + dyn_biases_1[1].sum()
        + dyn_biases_1[2].sum()
        + dyn_biases_1[3].sum()
    )
    total_loss.backward()

    # Verify gradients exist on all core parameters
    assert router.codebook_weights.grad is not None
    assert router.codebook_biases.grad is not None
    assert router.bias_head_down.weight.grad is not None
    assert router.bias_head_up.weight.grad is not None
    assert router.bias_head_up.bias.grad is not None
    assert router.embed_T.weight.grad is not None
    assert router.embed_P.weight.grad is not None
    assert router.king_embed.embed_chebyshev.weight.grad is not None


def test_dual_router_eval_mode_and_bit_packing():
    """Verify bit packing into uint16 and consistency in eval mode."""
    torch.manual_seed(42)
    router = DualVQCodebookRouter()
    router.eval()

    lut = router.get_lookup_table()
    assert lut.shape == (ROUTER_TABLE_SIZE,)

    c_sample = torch.tensor([0, 12345, 810809], dtype=torch.long)
    packed = lut[c_sample]

    e_w = packed & 0x3F
    e_b = (packed >> 6) & 0xFF

    assert all(0 <= w.item() < NUM_WEIGHT_EXPERTS for w in e_w)
    assert all(0 <= b.item() < NUM_BIAS_EXPERTS for b in e_b)
    assert get_router_hash(NUM_WEIGHT_EXPERTS, NUM_BIAS_EXPERTS) == (
        ROUTER_HASH ^ ((NUM_BIAS_EXPERTS << 16) | NUM_WEIGHT_EXPERTS)
    )

    coalesced_biases = router.get_coalesced_bias_table()
    assert coalesced_biases.shape == (NUM_BIAS_EXPERTS, NUM_BIASES)

    e1_w, e2_w, dyn_b1, dyn_b2, *_ = router(c_sample, c_sample)
    assert torch.equal(e1_w, e_w)
    assert torch.equal(e2_w, e_w)
    assert torch.equal(dyn_b1[0], coalesced_biases[e_b, 0:32])
    assert torch.equal(dyn_b1[1], coalesced_biases[e_b, 32:64])
    assert torch.equal(dyn_b1[2], coalesced_biases[e_b, 64:96])
    assert torch.equal(dyn_b1[3], coalesced_biases[e_b, 96:128])
    assert torch.equal(dyn_b2[0], coalesced_biases[e_b, 0:32])
    assert torch.equal(dyn_b2[1], coalesced_biases[e_b, 32:64])
    assert torch.equal(dyn_b2[2], coalesced_biases[e_b, 64:96])
    assert torch.equal(dyn_b2[3], coalesced_biases[e_b, 96:128])


def test_dead_expert_reseeding():
    """Verify that dead experts are identified and reseeded across both codebooks."""
    torch.manual_seed(99)
    router = DualVQCodebookRouter()

    # Force expert 0 in weights and expert 0 in biases to be extreme outliers / dead
    with torch.no_grad():
        router.codebook_weights[0].fill_(1000.0)
        router.codebook_biases[0].fill_(1000.0)
        router.ema_traffic_weights[0] = 0.0
        router.ema_traffic_biases[0] = 0.0

    reseeded_w, reseeded_b = router.reseed_dead_experts_context_wide()
    assert 0 in reseeded_w
    assert 0 in reseeded_b
    assert not torch.allclose(router.codebook_weights[0], torch.tensor(1000.0))
    assert not torch.allclose(router.codebook_biases[0], torch.tensor(1000.0))


def test_serialization_and_deserialization_roundtrip():
    """Verify that NNUEWriter and NNUEReader serialize and restore models identically."""
    torch.manual_seed(42)
    config = ModelConfig()
    model = NNUEModel(feature_name=DEFAULT_FEATURES, config=config)
    init_random_weights(model)

    # Initialize bias head with non-trivial values
    with torch.no_grad():
        model.router.bias_head_down.weight.normal_(0, 0.05)
        model.router.bias_head_up.weight.normal_(0, 0.05)
        model.router.bias_head_up.bias.normal_(0, 0.05)

    writer = NNUEWriter(model, verbose=False)
    serialized_bytes = writer.buf

    # Read back
    buf_io = io.BytesIO(serialized_bytes)
    reader = NNUEReader(buf_io, feature_name=DEFAULT_FEATURES, config=config)
    loaded_model = reader.model

    # Verify lookup tables match identically
    lut_orig = model.router.get_lookup_table()
    lut_loaded = loaded_model.router.get_lookup_table()
    assert torch.equal(lut_orig, lut_loaded)

    # Verify coalesced biases match within quantization precision
    coalesced_orig = model.router.get_coalesced_bias_table()
    coalesced_loaded = loaded_model.router.get_coalesced_bias_table()
    max_bias_diff = torch.max(torch.abs(coalesced_orig - coalesced_loaded)).item()
    assert max_bias_diff < 0.01, f"Dynamic bias diff too large: {max_bias_diff}"


def test_cpp_data_loader_parity():
    """Verify that C++ shared library produces valid and consistent context indices matching Python."""
    import chess

    from data_loader.stream import destroy_sparse_batch, get_sparse_batch_from_fens
    from tests.test_top2moe_symmetry import get_context_indices

    test_fens = [
        chess.STARTING_FEN,
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "8/4k3/1pbn1p1p/5Pp1/pP4P1/P2BK3/1B5P/8 w - - 1 49",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ]

    scores = [0] * len(test_fens)
    plies = [0] * len(test_fens)
    results = [0] * len(test_fens)

    batch_ptr = get_sparse_batch_from_fens(
        "HalfKAv2_hm", test_fens, scores, plies, results
    )
    assert batch_ptr is not None

    batch = batch_ptr.contents
    *_, c_us, c_them = batch.get_tensors("cpu")

    for i, fen in enumerate(test_fens):
        board = chess.Board(fen)
        expected_c_us, expected_c_them = get_context_indices(board)
        assert c_us[i].item() == expected_c_us, (
            f"FEN {fen}: C++ c_us={c_us[i].item()} != Python {expected_c_us}"
        )
        assert c_them[i].item() == expected_c_them, (
            f"FEN {fen}: C++ c_them={c_them[i].item()} != Python {expected_c_them}"
        )

    destroy_sparse_batch(batch_ptr)
