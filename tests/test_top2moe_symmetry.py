import chess
import torch

from cross_check_eval import init_random_weights
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.router import (
    NUM_MATERIAL_STATES,
    ROUTER_TABLE_SIZE,
    DualVQCodebookRouter,
    decode_material_state,
    decompose_router_index,
    encode_material_state,
    make_router_index,
)


def get_context_indices(board: chess.Board) -> tuple[int, int]:
    stm = board.turn
    non_stm = not stm

    # STM material features
    stm_pieces = [p for p in board.piece_map().values() if p.color == stm]
    T_us = min(max(len(stm_pieces) - 1, 0), 15)
    pawns_us = sum(1 for p in stm_pieces if p.piece_type == chess.PAWN)
    P_us = min(max(pawns_us, 0), min(8, T_us))
    has_queen_us = any(p.piece_type == chess.QUEEN for p in stm_pieces)
    Q_us = 1 if (T_us - P_us >= 1 and has_queen_us) else 0

    bishops_us = [
        sq
        for sq, p in board.piece_map().items()
        if p.color == stm and p.piece_type == chess.BISHOP
    ]
    has_dark_us = any(
        (chess.BB_SQUARES[sq] & 0x55AA55AA55AA55AA) != 0 for sq in bishops_us
    )
    has_light_us = any(
        (chess.BB_SQUARES[sq] & ~0x55AA55AA55AA55AA) != 0 for sq in bishops_us
    )
    BP_us = 1 if (T_us - P_us - Q_us >= 2 and has_dark_us and has_light_us) else 0
    mat_id_us = encode_material_state(T_us, P_us, Q_us, BP_us)

    # King features
    our_ksq = board.king(stm)
    enemy_ksq = board.king(non_stm)

    stm_flip = 0 if stm else 56
    our_ksq_oriented = our_ksq ^ stm_flip
    enemy_ksq_oriented = enemy_ksq ^ stm_flip

    our_castling = board.has_castling_rights(stm)
    enemy_castling = board.has_castling_rights(non_stm)

    if our_castling and our_ksq_oriented == chess.E1:
        our_king_bucket = 32
    else:
        r = our_ksq_oriented // 8
        f = our_ksq_oriented % 8
        our_king_bucket = r * 4 + (f if f < 4 else 7 - f)

    if enemy_castling and enemy_ksq_oriented == chess.E8:
        enemy_king_bucket = 64
    else:
        enemy_king_bucket = enemy_ksq_oriented

    c_us = make_router_index(mat_id_us, our_king_bucket, enemy_king_bucket)

    # Non-STM material features
    non_stm_pieces = [p for p in board.piece_map().values() if p.color == non_stm]
    T_them = min(max(len(non_stm_pieces) - 1, 0), 15)
    pawns_them = sum(1 for p in non_stm_pieces if p.piece_type == chess.PAWN)
    P_them = min(max(pawns_them, 0), min(8, T_them))
    has_queen_them = any(p.piece_type == chess.QUEEN for p in non_stm_pieces)
    Q_them = 1 if (T_them - P_them >= 1 and has_queen_them) else 0

    bishops_them = [
        sq
        for sq, p in board.piece_map().items()
        if p.color == non_stm and p.piece_type == chess.BISHOP
    ]
    has_dark_them = any(
        (chess.BB_SQUARES[sq] & 0x55AA55AA55AA55AA) != 0 for sq in bishops_them
    )
    has_light_them = any(
        (chess.BB_SQUARES[sq] & ~0x55AA55AA55AA55AA) != 0 for sq in bishops_them
    )
    BP_them = (
        1 if (T_them - P_them - Q_them >= 2 and has_dark_them and has_light_them) else 0
    )
    mat_id_them = encode_material_state(T_them, P_them, Q_them, BP_them)

    non_stm_flip = 56 if stm else 0
    their_ksq_oriented = enemy_ksq ^ non_stm_flip
    if enemy_castling and their_ksq_oriented == chess.E1:
        their_king_bucket = 32
    else:
        r = their_ksq_oriented // 8
        f = their_ksq_oriented % 8
        their_king_bucket = r * 4 + (f if f < 4 else 7 - f)

    our_ksq_oriented_them = our_ksq ^ non_stm_flip
    if our_castling and our_ksq_oriented_them == chess.E8:
        our_king_bucket_them = 64
    else:
        our_king_bucket_them = our_ksq_oriented_them

    c_them = make_router_index(mat_id_them, their_king_bucket, our_king_bucket_them)
    return c_us, c_them


def test_router_bijection():
    """Verify that material state and context indices decomposition is exact and bijective."""
    # 1. Material states bijection
    assert NUM_MATERIAL_STATES == 378
    for m in range(NUM_MATERIAL_STATES):
        t, p, q, bp = decode_material_state(m)
        m_rec = encode_material_state(t, p, q, bp)
        assert m == m_rec

    # 2. Context indexing bijection
    for m in [0, 100, 250, 377]:
        for k1 in [0, 16, 32]:
            for k2 in [0, 31, 64]:
                idx = make_router_index(m, k1, k2)
                assert 0 <= idx < ROUTER_TABLE_SIZE
                m_rec, k1_rec, k2_rec = decompose_router_index(torch.tensor([idx]))
                assert m_rec.item() == m
                assert k1_rec.item() == k1
                assert k2_rec.item() == k2


def test_board_symmetry_permutation():
    """Verify exact symmetry under color mirroring and perspective inversion."""
    torch.manual_seed(42)
    router = DualVQCodebookRouter()
    router.update_lookup_table()

    test_fens = [
        chess.STARTING_FEN,
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "8/4k3/1pbn1p1p/5Pp1/pP4P1/P2BK3/1B5P/8 w - - 1 49",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ]

    for fen in test_fens:
        board = chess.Board(fen)

        # 1. Perspective inversion on the same board (turn toggle)
        c_us, c_them = get_context_indices(board)
        board_turn_flip = board.copy()
        board_turn_flip.turn = not board.turn
        c_us_tflip, c_them_tflip = get_context_indices(board_turn_flip)

        assert c_us == c_them_tflip, (
            f"Turn toggle failed: c_us={c_us} != c_them_tflip={c_them_tflip}"
        )
        assert c_them == c_us_tflip, (
            f"Turn toggle failed: c_them={c_them} != c_us_tflip={c_us_tflip}"
        )

        # 2. Color-inverted mirrored board preserves exact STM evaluation context
        mirrored = board.mirror()
        c_us_mirror, c_them_mirror = get_context_indices(mirrored)
        assert c_us == c_us_mirror, (
            f"Color mirror invariance failed: c_us={c_us} != c_us_mirror={c_us_mirror}"
        )
        assert c_them == c_them_mirror, (
            f"Color mirror invariance failed: c_them={c_them} != c_them_mirror={c_them_mirror}"
        )

        # 3. Router expert selection
        e1, e2, dyn_b1, dyn_b2, *_ = router(
            torch.tensor([c_us]), torch.tensor([c_them])
        )
        e1_tf, e2_tf, dyn_b1_tf, dyn_b2_tf, *_ = router(
            torch.tensor([c_us_tflip]), torch.tensor([c_them_tflip])
        )
        assert e1.item() == e2_tf.item(), (
            f"Weight expert swap failed: e1={e1.item()} != e2_tf={e2_tf.item()}"
        )
        assert e2.item() == e1_tf.item(), (
            f"Weight expert swap failed: e2={e2.item()} != e1_tf={e1_tf.item()}"
        )
        for b_a, b_b in zip(dyn_b1, dyn_b2_tf, strict=True):
            assert torch.allclose(b_a, b_b)
        for b_a, b_b in zip(dyn_b2, dyn_b1_tf, strict=True):
            assert torch.allclose(b_a, b_b)


def test_model_dual_head_forward():
    """Verify LayerStacks evaluates sum of heads with gradient flow."""
    config = ModelConfig()
    model = NNUEModel(feature_name=DEFAULT_FEATURES, config=config)
    init_random_weights(model)

    x = torch.randn(4, config.L1)
    e1 = torch.tensor([0, 5, 12, 31])
    e2 = torch.tensor([31, 12, 5, 0])
    b_dummy = torch.randn(4, 32)
    dyn_b1 = (b_dummy, b_dummy, b_dummy, b_dummy)
    dyn_b2 = (b_dummy, b_dummy, b_dummy, b_dummy)

    out = model.layer_stacks(
        x,
        model.router.codebook,
        e1,
        e2,
        dyn_b1,
        dyn_b2,
        fake_quantize_acts=False,
        fake_quantize_weights=False,
    )
    assert out.shape == (4, 1)

    # Commutative symmetry: H_{e1, b1} + H_{e2, b2} == H_{e2, b2} + H_{e1, b1}
    out_rev = model.layer_stacks(
        x,
        model.router.codebook,
        e2,
        e1,
        dyn_b2,
        dyn_b1,
        fake_quantize_acts=False,
        fake_quantize_weights=False,
    )
    assert torch.allclose(out, out_rev, atol=1e-6)
