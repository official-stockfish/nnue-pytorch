"""Independent global feature reference and training-free contract checks."""

import io
import json
import os
import random
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import chess
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import stream
from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.features.global_features import GlobalSupports
from model.utils.serialize import NNUEReader, NNUEWriter

FENS = [
    chess.STARTING_FEN,
    "4k3/8/8/4Pp2/8/8/8/4K3 w - f6 0 1",
    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "4k3/8/2p5/2pP4/2P5/2P5/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
    "r3k2r/ppp2ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 8",
]
TARGETS = [
    [-1, 0, -1, 1, -1],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, -1],
    [0, 1, 2, 3, -1],
    [0, 1, 2, 3, 4],
]
BASES = [0, 0, 168, 1848, 4088, 7672]
SLOTS = [0, 84, 336, 560, 896, 1456]


def pseudo(pt, square, white=True):
    if pt == chess.PAWN:
        return chess.BB_PAWN_ATTACKS[white][square]
    if pt == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[square]
    diagonal = chess.BB_DIAG_ATTACKS[square][0]
    straight = chess.BB_RANK_ATTACKS[square][0] | chess.BB_FILE_ATTACKS[square][0]
    return (
        diagonal
        if pt == chess.BISHOP
        else straight
        if pt == chess.ROOK
        else diagonal | straight
    )


PREFIX = {pt: [] for pt in range(1, 6)}
for _pt in range(1, 6):
    _offset = 0
    for _square in range(64):
        PREFIX[_pt].append(_offset)
        if _pt != chess.PAWN or 8 <= _square < 56:
            _offset += pseudo(_pt, _square).bit_count()
    assert _offset == SLOTS[_pt]


def reference_relations(board, support, perspective):
    result = []
    for source, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        for target in board.attacks(source):
            other = board.piece_at(target)
            if other is None or other.piece_type == chess.KING:
                continue
            pt, target_pt = piece.piece_type, other.piece_type
            if (piece.color == other.color) != support or (
                pt == target_pt and source < target
            ):
                continue
            slot = TARGETS[pt - 1][target_pt - 1]
            if slot < 0:
                continue
            a, b, owner = source, target, piece.color
            reflected = not owner if support else not perspective
            if reflected:
                a ^= 56
                b ^= 56
            color = True if support else owner == perspective
            if pt == target_pt and a < b:
                a, b = b, a
                if not support:
                    color = not color
            row = BASES[pt] + slot * SLOTS[pt] + PREFIX[pt][a]
            row += (pseudo(pt, a, color) & ((1 << b) - 1)).bit_count()
            if support:
                row += 14952 * (owner != perspective)
            else:
                row += 14952 * (not color)
            result.append(row)
    return sorted(result)


def reference_pawns(board, perspective):
    states, pairs = [], []
    for file in range(8):
        pawns = sorted(
            (s, p.color)
            for s, p in board.piece_map().items()
            if p.piece_type == chess.PAWN and chess.square_file(s) == file
        )
        w = max((chess.square_rank(s) for s, c in pawns if c), default=0)
        b = min((chess.square_rank(s) for s, c in pawns if not c), default=0)
        if not perspective:
            w, b = 7 - b if b else 0, 7 - w if w else 0
        states.append(7 * w + b)
        # Enumerating all legal endpoint pairs independently of the native formula
        # fixes the serialized order, including unused states.
        vocabulary = [(0, a, 0, b) for a in range(6) for b in range(a + 1, 6)]
        vocabulary += [(1, a, 1, b) for a in range(6) for b in range(a + 1, 6)]
        vocabulary += [(0, a, 1, b) for a in range(6) for b in range(6) if a != b]
        for first, second in pairwise(pawns):
            endpoints = []
            for square, color in (first, second):
                rank = chess.square_rank(square) - 1
                if not perspective:
                    color, rank = not color, 5 - rank
                endpoints.append((int(not color), rank))
            endpoints.sort()
            (c1, r1), (c2, r2) = endpoints
            pairs.append(60 * file + vocabulary.index((c1, r1, c2, r2)))
    return [2401 * f + 49 * states[f] + states[f + 1] for f in range(7)], sorted(pairs)


def native_ids(component, fens):
    batch = stream.get_sparse_batch_from_fens(
        component, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    try:
        tensors = batch.contents.get_tensors("cpu")
        return [
            [sorted(row[row >= 0].tolist()) for row in tensor]
            for tensor in tensors[2:4]
        ]
    finally:
        stream.destroy_sparse_batch(batch)


def reference_position(fen):
    board = chess.Board(fen)
    out = {}
    for color, suffix in [(chess.WHITE, "w"), (chess.BLACK, "b")]:
        out[f"support_{suffix}"] = reference_relations(board, True, color)
        out[f"attack_{suffix}"] = reference_relations(board, False, color)
        front, pairs = reference_pawns(board, color)
        out[f"pawn_{suffix}"] = sorted(front + [16807 + i for i in pairs])
    return out


def check_features(fens):
    expected = [reference_position(fen) for fen in fens]
    components = [
        ("Global_Supports", "support"),
        ("Global_Attacks", "attack"),
        ("Pawn_FrontierPairs", "front"),
        ("Pawn_FilePairs", "file"),
    ]
    for name, family in components:
        white, black = native_ids(name, fens)
        for i, fen in enumerate(fens):
            for suffix, actual in [("w", white[i]), ("b", black[i])]:
                if family in ("support", "attack"):
                    want = expected[i][f"{family}_{suffix}"]
                else:
                    front, pairs = reference_pawns(chess.Board(fen), suffix == "w")
                    want = sorted(front if family == "front" else pairs)
                assert actual == want, (name, suffix, fen, actual, want)
    return expected


def test_native_features_and_perspectives():
    rng = random.Random(42)
    board = chess.Board()
    fens = FENS.copy()
    for ply in range(120):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        if ply % 5 == 0:
            fens += [board.fen(), board.mirror().fen()]
    check_features(fens)


def test_support_virtual_rows_and_gradients():
    feature = GlobalSupports(16)
    weights = feature.merged_weight()
    assert weights.shape == (29904, 16)
    assert feature.weight.shape == (14952, 16)
    assert torch.equal(weights[:14952], -weights[14952:])
    loss = weights[7].sum() + 3 * weights[14952 + 7].sum()
    loss.backward()
    assert torch.equal(feature.weight.grad[7], torch.full((16,), -2.0))
    assert torch.count_nonzero(feature.weight.grad).item() == 16


def test_export_round_trip_and_permutation():
    from ftperm import ft_permute_impl

    torch.manual_seed(42)
    config = ModelConfig(L1=32)
    model = NNUEModel(DEFAULT_FEATURES, config)
    model.clip_weights(include_input=True)
    encoded = NNUEWriter(model, verbose=False).buf
    restored = NNUEReader(io.BytesIO(encoded), DEFAULT_FEATURES, config).model
    assert NNUEWriter(restored, verbose=False).buf == encoded
    support = restored.input.features[0]
    assert torch.equal(
        support.merged_weight()[:14952], -support.merged_weight()[14952:]
    )
    original = support.weight.clone()
    permutation = torch.randperm(16).numpy()
    ft_permute_impl(restored, permutation)
    ft_permute_impl(restored, permutation.argsort())
    assert torch.equal(original, support.weight)


def test_old_format_rejected():
    model = NNUEModel(DEFAULT_FEATURES, ModelConfig(L1=32))
    data = NNUEWriter(model, verbose=False).buf
    data[:4] = (0x6A448AFA).to_bytes(4, "little")
    with pytest.raises((ValueError, AssertionError)):
        NNUEReader(io.BytesIO(data), DEFAULT_FEATURES, ModelConfig(L1=32))


def test_engine_feature_parity(tmp_path):
    executable = os.environ.get("GLOBAL_NNUE_TEST")
    if executable is None:
        pytest.skip("Set GLOBAL_NNUE_TEST to the Stockfish global-nnue-test executable")
    positions = tmp_path / "positions.epd"
    positions.write_text("\n".join(FENS) + "\n")
    output = subprocess.check_output([executable, "--dump", str(positions)], text=True)
    assert [json.loads(line) for line in output.splitlines()] == check_features(FENS)


if __name__ == "__main__":
    # Larger local/CI parity check without requiring a training dataset.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", required=True)
    parser.add_argument("--engine-test", required=True)
    args = parser.parse_args()
    fens = Path(args.positions).read_text().splitlines()
    output = subprocess.check_output(
        [args.engine_test, "--dump", args.positions], text=True
    )
    engine = [json.loads(line) for line in output.splitlines()]
    expected = []
    for offset in range(0, len(fens), 256):
        expected.extend(check_features(fens[offset : offset + 256]))
    assert engine == expected
    print(f"PASS engine/loader/Python parity: {len(fens)} positions")
