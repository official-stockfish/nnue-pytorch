"""

NOTE: This script uses CUDA and may requires large amounts of VRAM. Decrease --count if encountering problems.

Example use:

1. Generate the activation matrix for some sample dataset.

python ftperm.py gather --data=data\fishpack32.binpack --net=networks\nn-5af11540bbfe.nnue --count=1000000 --features=HalfKAv2_hm --out ftact1m.npy

python ftperm.py gather --data=noob_master_leaf_static_d12_85M_0.binpack --net=nn-5af11540bbfe.nnue --count=10000 --features=HalfKAv2_hm --out ftact1m.npy

2. Find a permutation

python ftperm.py find_perm --data=ftact1m.npy --out=ftact.perm

3. Test the permutation against the baseline

python ftperm.py eval_perm --data=ftact1m.npy --perm=ftact.perm

4. Apply permutation and save
python serialize.py nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm --ft_perm=ftact.perm

----------------------------------------------------------------

OR do the whole process in one step

python serialize.py networks\nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm --ft_optimize --ft_optimize_data=data\fishpack32.binpack --ft_optimize_count=1000000

python serialize.py nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm --ft_optimize --ft_optimize_data=noob_master_leaf_static_d12_85M_0.binpack --ft_optimize_count=10000

"""

import time
import argparse
import features
import serialize
import nnue_dataset
import chess
import model as M
import torch
import copy
import numpy as np
from model import NNUE
import cupy as cp

"""

Algorithm by Daniel Monroe. Github @Ergodice.

"""

ZERO_BLOCK_SIZE = 4
VERBOSE = False
USE_CUPY = False


def batched(arr, batch_size):
    """
    Utility generator that yields chunks of array `arr` of size `batch_size`
    Expects arr to be a numpy-like array
    """
    n_samples = arr.shape[0]
    idx = 0
    while idx < n_samples:
        yield arr[idx : min(idx + batch_size, n_samples)]
        idx += batch_size


def apply_swap(perm, i, j):
    """
    Swap `i`-th and `j`-th elements in the array `perm`.
    """
    perm[i], perm[j] = perm[j], perm[i]


def apply_rotate_right(perm, indices):
    """
    Rotates right the values in `perm` at selected indices `indices`.
    The rotation is performed as-if the selected indices were layed out in the order
    specified in the `indices` list.
    """
    values = [perm[i] for i in indices]
    new_values = [values[-1]] + values[:-1]
    for i, j in zip(indices, new_values):
        perm[i] = j


def get_swapped_zero_positive_count(actmat_flat, use_cupy=True):
    if use_cupy:
        actmat_flat = cp.asarray(actmat_flat, dtype=cp.int8)

    shape = actmat_flat.shape
    # Group into blocks that are processed at once during inference
    # actmat is a boolean matrix of shape (N, L1 // 2) with "True" meaning 0
    actmat_chunked = actmat_flat.reshape(
        (actmat_flat.shape[0], actmat_flat.shape[1] // ZERO_BLOCK_SIZE, ZERO_BLOCK_SIZE)
    )

    if use_cupy:
        # Calculate number of zeros in each block
        num_zeros = cp.sum(actmat_chunked, axis=2, keepdims=True)
        # Broadcast back to the same shape as actmat_chunked so it's easier to work with
        num_zeros = cp.tile(num_zeros, (1, 1, ZERO_BLOCK_SIZE))

        # Marks an element if all other elements in a block are zero.
        #
        # Example:
        #                                   b  i   k      b  i   k      b  i   k
        # slice                            [0, 13, :]    [0, 14, :]    [0, 15, :]
        # num_zeros           = [... [... [3, 3, 3, 3], [1, 1, 1, 1], [4, 4, 4, 4] ...] ...]
        # actmat_chunked      = [... [... [1, 1, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1] ...] ...]
        # rest_zero_indicator = [... [... [0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1] ...] ...]
        #
        rest_zero_indicator = (
            (num_zeros - actmat_chunked == ZERO_BLOCK_SIZE - 1)
            .reshape(shape)
            .astype(cp.int8)
        )

        # Sum all possible pairs of elements in a single sample of actmat_flat and rest_zero_indicator.
        # Aggregate sum over the whole batch.
        # This tells us how much "good" a swap of i-th and j-th slices would do. It doesn't consider
        # how much "bad" it would do though, that will be accounted for later, for performance reasons.
        swapped_zero_count = cp.einsum(
            "bi,bj->ij", actmat_flat, rest_zero_indicator, dtype=int
        )

    else:
        # Same operation but with numpy
        num_zeros = np.sum(actmat_chunked, axis=2, keepdims=True)
        num_zeros = np.tile(num_zeros, (1, 1, ZERO_BLOCK_SIZE))

        rest_zero_indicator = (
            (num_zeros - actmat_chunked == ZERO_BLOCK_SIZE - 1)
            .reshape(shape)
            .astype(int)
        )

        swapped_zero_count = np.einsum("bi,bj->ij", actmat_flat, rest_zero_indicator)

    return swapped_zero_count


def get_swapped_zero_increase(actmat, use_cupy=True):
    n_neurons = actmat.shape[1]
    swapped_zero_count = 0

    # Process in batches since the arrays are too large
    # TODO: Find a good batch size. Try lowest as possible as VRAM is an issue on low end devices.
    BATCH_SIZE = 10000
    for actmat_batch in batched(actmat, BATCH_SIZE):
        swapped_zero_count += get_swapped_zero_positive_count(
            actmat_batch, use_cupy=use_cupy
        )

    # (L1/2) x (L1/2)
    if use_cupy:
        # Subtract from each i-th slice the positive value of the current i-th placement.
        # This is the place where we account for how much "bad" it would do.
        # It is done here because we process earlier in batches, but this operation is distributive,
        # so it needs to only be done once at the end.
        swapped_zero_increase = swapped_zero_count - cp.reshape(
            cp.diag(swapped_zero_count), (1, n_neurons)
        )
        swapped_zero_increase = cp.asnumpy(swapped_zero_increase)

    else:
        swapped_zero_increase = swapped_zero_count - np.reshape(
            np.diag(swapped_zero_count), (1, n_neurons)
        )

    return swapped_zero_increase


def get_score_change(actmat, use_cupy=True):
    # actmat is a boolean matrix of shape (N, L1) with "True" meaning 0

    n_neurons = actmat.shape[1]

    score_change = get_swapped_zero_increase(actmat, use_cupy)

    # Kill off swaps between neurons in the same block
    blocks = np.arange(n_neurons).reshape((n_neurons, 1)) // ZERO_BLOCK_SIZE
    same_block_killer = 1 - (blocks == blocks.T).astype(int)
    score_change = score_change * same_block_killer
    return score_change


def make_swaps_2(actmat, use_cupy=True):
    """
    Returns a series of independent 2-swap operations that collectively improve the objective function.
    """

    # For each pair of nodes, we want to calculate the difference between the number of 4-zero runs when swapping them
    start_time = time.time()
    print("Starting make_swaps_2")

    n_neurons = actmat.shape[1]
    n_samples = actmat.shape[0]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE

    # Compute the score change of swapping i-th and j-th neurons
    score_change = get_score_change(actmat, use_cupy=use_cupy)
    # Sum score_change[i, j] + score_change[j, i] to get the cumulative impact of the swap.
    score_change = score_change + score_change.T

    def all_indices_in_same_block(i):
        """Returns a list of indices of all neurons in the same block as the i-th neuron."""
        # Floor to the start of the block.
        base = i // ZERO_BLOCK_SIZE * ZERO_BLOCK_SIZE
        return list(range(base, base + ZERO_BLOCK_SIZE))

    swaps = []
    total_score_change = 0
    while True:
        swap = np.argmax(score_change)
        # argmax returns a flat index, so we need to recompute the position.
        i, j = swap // n_neurons, swap % n_neurons

        improvement = score_change[i, j]
        if improvement == 0:
            break

        if VERBOSE:
            print(f"Swapping {i} and {j} for improvement {improvement}")

        # The swap is an improvement, add it to the list.
        total_score_change += improvement
        swaps.append((i, j))

        indices_to_kill = all_indices_in_same_block(i) + all_indices_in_same_block(j)
        for index in indices_to_kill:
            # Zero out the improvement for the swaps to and from blocks which had neurons swapped.
            # This ensures they won't be picked later, and therefore all swaps will be independent.
            score_change[:, index] = 0
            score_change[index, :] = 0

    total_improvement = (
        total_score_change / n_samples / (n_neurons // ZERO_BLOCK_SIZE) * 100
    )

    print(f"Time elapsed: {time.time() - start_time:0.3f}")
    print(f"Improvement this iteration: {total_improvement:0.3f}")

    return swaps, total_improvement


def make_swaps_3(actmat, use_cupy=True):
    """
    Returns a series of independent left-rotates operations that collectively improve the objective function.
    """

    # For each triplet of nodes, we want to calculate the change in score when moving them in a cycle
    print("Starting make_swaps_3")
    start_time = time.time()

    n_neurons = actmat.shape[1]
    n_samples = actmat.shape[0]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE

    score_changes = get_score_change(actmat, use_cupy=use_cupy)

    # For each neuron i, j, k we sum score_change[i, j] + score_change[j, k] + score_change[k, i]
    # This is the cumulative impact of the right-rotation.
    score_changes = (
        score_changes[:, :, None]
        + score_changes[None, :, :]
        + (score_changes.T)[:, None, :]
    )

    orig_shape = (n_neurons,) * 3
    compressed_shape = (n_blocks, ZERO_BLOCK_SIZE) * 3
    cycles = []
    total_score_change = 0

    if use_cupy:
        # We don't want to have to go through an enormous array so compress it to represent blocks rather than neurons
        # Cupy doesn't support a list of axes so we go one by one.
        max_values = cp.amax(
            cp.reshape(score_changes, compressed_shape), axis=5, keepdims=False
        )
        max_values = cp.amax(max_values, axis=3, keepdims=False)
        max_values = cp.amax(max_values, axis=1, keepdims=False)
    else:
        max_values = np.amax(
            np.reshape(score_changes, compressed_shape), axis=(5, 3, 1), keepdims=False
        )

    # Kill rotates that would only affect less than 3 different blocks.
    # We must do this, because the rest of the algorithm relies on it for correctness.
    # It would also be pointless as such cases degenerate to the ones handled by make_swaps_2.
    for block in range(n_blocks):
        max_values[block, block, :] = 0
        max_values[block, :, block] = 0
        max_values[:, block, block] = 0

    while True:
        best_blocks = max_values.argmax()
        improvement_blocks = max_values.flatten()[best_blocks]
        if improvement_blocks == 0:
            break

        total_score_change += improvement_blocks

        # We first find the blocks that have neurons that can be rotated with a gain.
        b1, b2, b3 = np.unravel_index(best_blocks, (n_blocks, n_blocks, n_blocks))
        i, j, k = b1 * ZERO_BLOCK_SIZE, b2 * ZERO_BLOCK_SIZE, b3 * ZERO_BLOCK_SIZE

        # Now we need to find the best set of neurons for this rotation in the found blocks
        # (we already know there is a gain available)
        local_score_changes = score_changes[
            i : i + ZERO_BLOCK_SIZE, j : j + ZERO_BLOCK_SIZE, k : k + ZERO_BLOCK_SIZE
        ]
        best_neurons = local_score_changes.argmax()
        improvement_neurons = local_score_changes.flatten()[best_neurons]
        assert improvement_blocks == improvement_neurons
        i1, j1, k1 = np.unravel_index(
            best_neurons, (ZERO_BLOCK_SIZE, ZERO_BLOCK_SIZE, ZERO_BLOCK_SIZE)
        )
        i, j, k = i + i1, j + j1, k + k1

        if VERBOSE:
            print(f"Right-rotating {i}, {j}, {k} for improvement {improvement_neurons}")

        # Add the right-rotate indices. We add them in reverse order as we previously computed for a right-rotate.
        cycles.append((i, j, k))

        # Now silence these blocks since the scores are no longer accurate
        # We only need to affect the smaller array since gains of zeros and under are ignored
        for b in (b1, b2, b3):
            max_values[b, :, :] = 0
            max_values[:, b, :] = 0
            max_values[:, :, b] = 0

    total_improvement = total_score_change / n_samples / (n_neurons // 4) * 100
    print(f"Time elapsed: {time.time() - start_time:0.3f}")
    print(f"Improvement this iteration: {total_improvement:0.3f}")
    return cycles, total_improvement


def find_perm_impl(actmat):
    actmat = np.reshape(actmat, (actmat.shape[0] * 2, actmat.shape[1] // 2))
    if USE_CUPY:
        actmat = cp.asarray(actmat, dtype=cp.int8)
    actmat_orig = actmat.copy()

    total_score_change = 0
    perm = np.arange(M.L1 // 2)

    stages = [make_swaps_2, make_swaps_3]
    # The optimization routines are deterministic, so no need to retry.
    stages_max_fails = [0, 0]
    stage_id = 0
    stop_after_stage = None
    num_fails = 0

    for i in range(50):
        print("Iteration", i + 1)

        # Choose the current stage optimization function
        swap_fn = stages[stage_id]

        # Apply the current permutation to get the current best neuron order.
        actmat = actmat_orig[:, perm]

        # Calculate a set of independent right rotates (so swaps for 2 element case)
        # that when applied improve the objective function
        swaps, score_change = swap_fn(actmat, USE_CUPY)
        for cycle in swaps:
            # Update the current best permutation with the newly found adjustments.
            apply_rotate_right(perm, cycle)

        total_score_change += score_change
        print(f"Total improvement: {total_score_change}\n")

        if score_change == 0:
            num_fails += 1
            if num_fails > stages_max_fails[stage_id]:
                num_fails = 0
                stage_id += 1

                if stage_id >= len(stages) or (
                    stop_after_stage is not None and stage_id > stop_after_stage
                ):
                    print("No more improvement possible.")
                    break

                print(f"Switching to stage {stage_id}")

    return perm


# -------------------------------------------------------------


def read_model(nnue_path, feature_set):
    with open(nnue_path, "rb") as f:
        reader = serialize.NNUEReader(f, feature_set)
        return reader.model


def make_fen_batch_provider(data_path, batch_size):
    return nnue_dataset.FenBatchProvider(data_path, True, 1, batch_size, False, 10)


def filter_fens(fens):
    # We don't want fens where a king is in check, as these cannot be evaluated by the engine.
    filtered_fens = []
    for fen in fens:
        board = chess.Board(fen=fen)
        if not board.is_check():
            filtered_fens.append(fen)
    return filtered_fens


def quantize_ft(model):
    model.input.weight.data = model.input.weight.data.mul(model.quantized_one).round()
    model.input.bias.data = model.input.bias.data.mul(model.quantized_one).round()


def forward_ft(
    model,
    us,
    them,
    white_indices,
    white_values,
    black_indices,
    black_values,
    psqt_indices,
    layer_stack_indices,
):
    wp, bp = model.input(white_indices, white_values, black_indices, black_values)
    w, wpsqt = torch.split(wp, M.L1, dim=1)
    b, bpsqt = torch.split(bp, M.L1, dim=1)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    l0_ = torch.clamp(l0_, 0.0, 127.0)

    l0_s = torch.split(l0_, M.L1 // 2, dim=1)
    l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
    # We multiply by 127/128 because in the quantized network 1.0 is represented by 127
    # and it's more efficient to divide by 128 instead.
    l0_ = torch.cat(l0_s1, dim=1) * (1 / 128)

    return l0_.round()


def eval_ft(model, batch):
    with torch.no_grad():
        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        ) = batch.contents.get_tensors("cuda")
        res = forward_ft(
            model,
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            layer_stack_indices,
        )
        return res


def ft_permute_impl(model, permutation):
    permutation = list(permutation)

    l1_size = model.layer_stacks.l1.in_features
    if l1_size != len(permutation) * 2:
        raise Exception(
            f"Invalid permutation size. Expected {l1_size}. Got {len(permutation) * 2}."
        )

    # Both sides of the FT must use the same permutation.
    permutation.extend([x + l1_size // 2 for x in permutation])

    # Add identity permutation for PSQT weights
    ft_permutation = permutation + list(range(l1_size, model.input.num_outputs))

    # Apply the permutation in place.
    model.input.weight.data = model.input.weight.data[:, ft_permutation]
    model.input.bias.data = model.input.bias.data[ft_permutation]
    model.layer_stacks.l1.weight.data = model.layer_stacks.l1.weight.data[
        :, permutation
    ]


def ft_permute(model, ft_perm_path):
    with open(ft_perm_path, "rb") as f:
        permutation = np.load(f)

    ft_permute_impl(model, permutation)


def gather_impl(model, dataset, count):
    ZERO_POINT = 0.0  # Vary this to check hypothetical forced larger truncation to zero
    BATCH_SIZE = 1000

    old_device = model.device

    quantized_model = copy.deepcopy(model)
    quantize_ft(quantized_model)
    quantized_model.cuda()

    fen_batch_provider = make_fen_batch_provider(dataset, BATCH_SIZE)

    actmats = []

    done = 0
    print("Processed {} positions.".format(done))
    while done < count:
        fens = filter_fens(next(fen_batch_provider))

        b = nnue_dataset.make_sparse_batch_from_fens(
            quantized_model.feature_set,
            fens,
            [0] * len(fens),
            [1] * len(fens),
            [0] * len(fens),
        )
        actmat = eval_ft(quantized_model, b).cpu()
        actmat = actmat <= ZERO_POINT
        actmats.append(actmat.numpy())
        nnue_dataset.destroy_sparse_batch(b)

        done += len(fens)
        print("Processed {} positions.".format(done))

    return np.concatenate(actmats, axis=0)


def command_gather(args):
    feature_set = features.get_feature_set_from_name(args.features)
    if args.checkpoint:
        model = NNUE.load_from_checkpoint(args.checkpoint, feature_set=feature_set)
    else:
        model = read_model(args.net, feature_set)

    model.eval()

    actmat = gather_impl(model, args.data, args.count)

    with open(args.out, "wb") as file:
        np.save(file, actmat)


def eval_act_mat(actmat):
    actmat = actmat.reshape((actmat.shape[0], actmat.shape[1] // 4, 4))
    r = np.all(actmat, axis=2)
    return np.count_nonzero(r) / r.shape[0] / r.shape[1]


def eval_perm_impl(actmat, perm=None):
    actmat = np.reshape(actmat, (actmat.shape[0] * 2, actmat.shape[1] // 2))

    actmat_eval = eval_act_mat(actmat)
    print(f"Combined zeros in base matrix: {actmat_eval * 100:0.6f}")

    if perm is not None:
        perm_act_mat = actmat[:, perm]
        perm_act_mat_eval = eval_act_mat(perm_act_mat)
        print(f"Combined zeros in perm matrix: {perm_act_mat_eval * 100:0.6f}")


def command_eval_perm(args):
    with open(args.data, "rb") as file:
        actmat = np.load(file)

    if args.perm is not None:
        with open(args.perm, "rb") as file:
            perm = np.load(file)
    else:
        perm = None

    eval_perm_impl(actmat, perm)


def command_find_perm(args):
    with open(args.data, "rb") as file:
        actmat = np.load(file)

    perm = find_perm_impl(actmat)

    # perm = np.random.permutation([i for i in range(M.L1)])
    with open(args.out, "wb") as file:
        np.save(file, perm)


def ft_optimize(model, dataset_path, count, actmat_save_path=None, perm_save_path=None):
    print("Gathering activation data...")
    actmat = gather_impl(model, dataset_path, count)
    if actmat_save_path is not None:
        with open(actmat_save_path, "wb") as file:
            np.save(file, actmat)

    print("Finding permutation...")
    perm = find_perm_impl(actmat)
    if actmat_save_path is not None:
        with open(perm_save_path, "wb") as file:
            np.save(file, perm)

    print("Evaluating permutation...")
    eval_perm_impl(actmat, perm)

    print("Applying permutation...")
    ft_permute_impl(model, perm)


def main():
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers()

    parser_gather = subparsers.add_parser("gather", help="a help")
    parser_gather.add_argument("--net", type=str, help="path to a .nnue net")
    parser_gather.add_argument(
        "--data", type=str, help="path to a .bin or .binpack dataset"
    )
    parser_gather.add_argument(
        "--checkpoint",
        type=str,
        help="Optional checkpoint (used instead of nnue for local eval)",
    )
    parser_gather.add_argument(
        "--count", type=int, default=1000, help="number of datapoints to process"
    )
    parser_gather.add_argument(
        "--out", type=str, help="Filename under which to save the resulting ft matrix"
    )
    features.add_argparse_args(parser_gather)
    parser_gather.set_defaults(func=command_gather)

    parser_gather = subparsers.add_parser("find_perm", help="a help")
    parser_gather.add_argument(
        "--data", type=str, help="path to the previously gathered ft activation data"
    )
    parser_gather.add_argument(
        "--out", type=str, help="path to where to save the permutation"
    )
    parser_gather.set_defaults(func=command_find_perm)

    parser_gather = subparsers.add_parser("eval_perm", help="a help")
    parser_gather.add_argument(
        "--data", type=str, help="path to the previously gathered ft activation data"
    )
    parser_gather.add_argument(
        "--perm", type=str, help="path to the previously generated perm file"
    )
    parser_gather.set_defaults(func=command_eval_perm)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
