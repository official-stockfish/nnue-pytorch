"""

NOTE: This script uses CUDA and may require large amounts of VRAM. Decrease --count if encountering problems.

Example use:

1. Generate the activation matrix for some sample dataset.

python ftperm.py gather --data=data\fishpack32.binpack --net=networks\nn-5af11540bbfe.nnue --count=1000000 --features=HalfKAv2_hm^ --out ftact1m.npy

python ftperm.py gather --data=noob_master_leaf_static_d12_85M_0.binpack --net=nn-5af11540bbfe.nnue --count=10000 --features=HalfKAv2_hm^ --out ftact1m.npy

2. Find a permutation

python ftperm.py find_perm --data=ftact1m.npy --out=ftact.perm

3. Test the permutation against the baseline

python ftperm.py eval_perm --data=ftact1m.npy --perm=ftact.perm

4. Apply permutation and save
python serialize.py nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm^ --ft_perm=ftact.perm

----------------------------------------------------------------

OR do the whole process in one step

python serialize.py networks\nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm^ --ft_optimize --ft_optimize_data=data\fishpack32.binpack --ft_optimize_count=1000000

python serialize.py nn-5af11540bbfe.nnue permuted.nnue --features=HalfKAv2_hm^ --ft_optimize --ft_optimize_data=noob_master_leaf_static_d12_85M_0.binpack --ft_optimize_count=10000

"""

import copy
from dataclasses import dataclass, field, replace
import time
from typing import Callable, Generator, TypeAlias, Annotated, Union, Literal, TypeVar, Optional

import tyro

from tyro.conf import (
    OmitArgPrefixes,
)

import numpy as np
import numpy.typing as npt
import torch

import data_loader
import model as M
from model import (
    NNUE,
    NNUEModel,
    NNUEReader,
    FeatureConfig,
    ModelConfig,
)


"""

Algorithm by Daniel Monroe. Github @Ergodice.

"""

ZERO_BLOCK_SIZE = 4
VERBOSE = False
_DEVICE_OVERRIDE = None

@dataclass
class GatherConfig:
    data: str
    """
    path to a .bin or .binpack dataset
    """
    out: str
    """
    Filename under which to save the resulting ft matrix
    """
    net: str | None = None
    """
    path to a .nnue net
    """
    checkpoint: str | None = None
    """
    Optional checkpoint (used instead of nnue for local eval)
    """
    count: int = 1000
    """
    number of datapoints to process (lower bound, actual count will be a multi of batch_size=1024)
    """
    loader_num_workers: int = 4
    """Number of workers to use for data loading during gathering FT activations."""

    loader_config: OmitArgPrefixes[data_loader.DataloaderSkipConfig] = field(
        default_factory=data_loader.DataloaderSkipConfig
    )
    feature_config: OmitArgPrefixes[FeatureConfig] = field(
        default_factory=FeatureConfig
    )


@dataclass
class FindPermConfig:
    data: str
    """
    path to the previously gathered ft activation data
    """
    out: str
    """
    path to where to save the permutation
    """


@dataclass
class EvalPermConfig:
    data: str
    """
    path to the previously gathered ft activation data
    """
    perm: str | None = None
    """
    path to the previously generated perm file"""


@dataclass
class FeaturePermutationConfig:
    subcommand: Annotated[
        Union[
            Annotated[
                GatherConfig,
                tyro.conf.subcommand("gather", prefix_name=False),
            ],
            Annotated[
                FindPermConfig,
                tyro.conf.subcommand("find_perm", prefix_name=False),
            ],
            Annotated[
                EvalPermConfig,
                tyro.conf.subcommand("eval_perm", prefix_name=False),
            ],
        ],
        tyro.conf.arg(name=""),
    ]
    use_cupy: Annotated[bool, tyro.conf.arg(name="cupy")] = True
    """
    Set to False to use CPU instead of GPU. Kept for legacy CLI compatibility.
    """
    device: Union[int, Literal["cpu", "mps"]] = 0
    """
    Device to use. Can be integer (e.g. 0 for cuda:0), "mps", or "cpu"
    """

    model_config: OmitArgPrefixes[ModelConfig] = field(default_factory=ModelConfig)


def resolve_device(use_cupy: bool, device: Union[int, Literal["cpu", "mps"]]) -> str:
    if not use_cupy:
        return "cpu"
    if _DEVICE_OVERRIDE is not None:
        d = str(_DEVICE_OVERRIDE)
    else:
        d = str(device)
    if d.isdigit():
        return f"cuda:{d}"
    return d


T = TypeVar("T", npt.NDArray, torch.Tensor)
def batched(arr: T, batch_size: int) -> Generator[T, None, None]:
    """
    Utility generator that yields chunks of array `arr` of size `batch_size`
    Expects arr to be a numpy-like array or torch Tensor
    """
    n_samples = arr.shape[0]
    idx = 0
    while idx < n_samples:
        yield arr[idx : min(idx + batch_size, n_samples)]
        idx += batch_size


def apply_swap(perm: npt.NDArray, i: int, j: int) -> None:
    """
    Swap `i`-th and `j`-th elements in the array `perm`.
    """
    perm[i], perm[j] = perm[j], perm[i]


def apply_rotate_right(perm: npt.NDArray, indices: tuple[int, ...]) -> None:
    """
    Rotates right the values in `perm` at selected indices `indices`.
    The rotation is performed as-if the selected indices were layed out in the order
    specified in the `indices` list.
    """
    values = [perm[i] for i in indices]
    new_values = [values[-1]] + values[:-1]
    for i, j in zip(indices, new_values):
        perm[i] = j


def get_swapped_zero_positive_count(
    actmat_flat: torch.Tensor
) -> torch.Tensor:
    shape = actmat_flat.shape
    # Group into blocks that are processed at once during inference
    # actmat is a boolean matrix of shape (N, L1 // 2) with "True" meaning 0
    actmat_chunked = actmat_flat.reshape(
        (shape[0], shape[1] // ZERO_BLOCK_SIZE, ZERO_BLOCK_SIZE)
    )

    # Calculate number of zeros in each block
    num_zeros = torch.sum(actmat_chunked, dim=2, keepdim=True)
    # Broadcast back to the same shape as actmat_chunked so it's easier to work with
    num_zeros = num_zeros.tile((1, 1, ZERO_BLOCK_SIZE))

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
        (num_zeros - actmat_chunked.int() == ZERO_BLOCK_SIZE - 1)
        .reshape(shape)
    )

    # Sum all possible pairs of elements in a single sample of actmat_flat and rest_zero_indicator.
    # Aggregate sum over the whole batch.
    # This tells us how much "good" a swap of i-th and j-th slices would do. It doesn't consider
    # how much "bad" it would do though, that will be accounted for later, for performance reasons.
    # Note: float32 has full precision up to a batch size of around 16M, more than enough for current cases.
    # int32 would offer full precision up to batch sizes of 2B instead.
    swapped_zero_count = (
        actmat_flat.to(torch.float32).T @ rest_zero_indicator.to(torch.float32)
    )

    return swapped_zero_count


def get_swapped_zero_increase(
    actmat: torch.Tensor
) -> torch.Tensor:
    n_neurons = actmat.shape[1]
    swapped_zero_count = 0

    # Process in batches since the arrays are too large
    BATCH_SIZE = 8192
    for actmat_batch in batched(actmat, BATCH_SIZE):
        swapped_zero_count += get_swapped_zero_positive_count(actmat_batch)

    # (L1/2) x (L1/2)
    # Subtract from each i-th slice the positive value of the current i-th placement.
    # This is the place where we account for how much "bad" it would do.
    # It is done here because we process earlier in batches, but this operation is distributive,
    # so it needs to only be done once at the end.
    swapped_zero_increase = swapped_zero_count - torch.reshape(
        torch.diag(swapped_zero_count), (1, n_neurons)
    )

    return swapped_zero_increase


def get_score_change(
    actmat: torch.Tensor
) -> torch.Tensor:
    # actmat is a boolean matrix of shape (N, L1) with "True" meaning 0

    n_neurons = actmat.shape[1]

    score_change = get_swapped_zero_increase(actmat)

    # Kill off swaps between neurons in the same block
    blocks = torch.arange(n_neurons, device=actmat.device).reshape((n_neurons, 1)) // ZERO_BLOCK_SIZE
    same_block_killer = 1 - (blocks == blocks.T).to(torch.int)
    score_change = score_change * same_block_killer
    return score_change


@dataclass
class SwapResult:
    swaps: list[tuple[int, ...]]
    score_change: float


SwapFunction: TypeAlias = Callable[[torch.Tensor], SwapResult]


def make_swaps_2(actmat: torch.Tensor) -> SwapResult:
    """
    Returns a series of independent 2-swap operations that collectively improve the objective function.
    """

    # For each pair of nodes, we want to calculate the difference between the number of 4-zero runs when swapping them
    start_time = time.time()
    print("Starting make_swaps_2")

    n_neurons = actmat.shape[1]
    n_samples = actmat.shape[0]

    # Compute the score change of swapping i-th and j-th neurons
    score_change = get_score_change(actmat)
    # Sum score_change[i, j] + score_change[j, i] to get the cumulative impact of the swap.
    score_change = score_change + score_change.T

    def all_indices_in_same_block(i: int) -> list[int]:
        """Returns a list of indices of all neurons in the same block as the i-th neuron."""
        # Floor to the start of the block.
        base = i // ZERO_BLOCK_SIZE * ZERO_BLOCK_SIZE
        return list(range(base, base + ZERO_BLOCK_SIZE))

    swaps = []
    total_score_change = 0
    while True:
        swap = torch.argmax(score_change).item()
        # argmax returns a flat index, so we need to recompute the position.
        i, j = swap // n_neurons, swap % n_neurons

        improvement = score_change[i, j].item()
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

    return SwapResult(swaps, total_improvement)


def make_swaps_3(actmat: torch.Tensor) -> SwapResult:
    """
    Returns a series of independent left-rotates operations that collectively improve the objective function.
    """

    # For each triplet of nodes, we want to calculate the change in score when moving them in a cycle
    print("Starting make_swaps_3")
    start_time = time.time()

    n_neurons = actmat.shape[1]
    n_samples = actmat.shape[0]
    n_blocks = n_neurons // ZERO_BLOCK_SIZE

    score_changes = get_score_change(actmat)

    # For each neuron i, j, k we sum score_change[i, j] + score_change[j, k] + score_change[k, i]
    # This is the cumulative impact of the right-rotation.
    score_changes = (
        score_changes[:, :, None]
        + score_changes[None, :, :]
        + (score_changes.T)[:, None, :]
    )

    compressed_shape = (n_blocks, ZERO_BLOCK_SIZE) * 3
    cycles = []
    total_score_change = 0

    # We don't want to have to go through an enormous array so compress it to represent blocks rather than neurons
    max_values = torch.amax(
        torch.reshape(score_changes, compressed_shape), dim=5, keepdim=False
    )
    max_values = torch.amax(max_values, dim=3, keepdim=False)
    max_values = torch.amax(max_values, dim=1, keepdim=False)

    # Kill rotates that would only affect less than 3 different blocks.
    # We must do this, because the rest of the algorithm relies on it for correctness.
    # It would also be pointless as such cases degenerate to the ones handled by make_swaps_2.
    for block in range(n_blocks):
        max_values[block, block, :] = 0
        max_values[block, :, block] = 0
        max_values[:, block, block] = 0

    while True:
        best_blocks = torch.argmax(max_values).item()
        improvement_blocks = max_values.flatten()[best_blocks].item()
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
        ].contiguous()
        best_neurons = torch.argmax(local_score_changes).item()
        improvement_neurons = local_score_changes.flatten()[best_neurons].item()
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
    return SwapResult(cycles, total_improvement)


def find_perm_impl(
    actmat: Union[npt.NDArray[np.bool_], torch.Tensor], device_str: str, L1: int
) -> npt.NDArray[np.int_]:
    if isinstance(actmat, np.ndarray):
        actmat = np.reshape(actmat, (actmat.shape[0] * 2, actmat.shape[1] // 2))
        actmat = torch.from_numpy(actmat).to(device_str)
    else:
        actmat = actmat.reshape((actmat.shape[0] * 2, actmat.shape[1] // 2))
        actmat = actmat.to(device_str)

    actmat_orig = actmat.clone()

    total_score_change = 0
    perm = np.arange(L1 // 2)

    stages: list[SwapFunction] = [make_swaps_2, make_swaps_3]
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
        swap_result = swap_fn(actmat)
        for cycle in swap_result.swaps:
            # Update the current best permutation with the newly found adjustments.
            apply_rotate_right(perm, cycle)

        total_score_change += swap_result.score_change
        print(f"Total improvement: {total_score_change}\n")

        if swap_result.score_change == 0:
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


def read_model(
    nnue_path: str,
    feature_name: str,
    config: M.ModelConfig,
) -> NNUEModel:
    with open(nnue_path, "rb") as f:
        reader = NNUEReader(f, feature_name, config)
        return reader.model


def make_sparse_batch_provider(
    data_path: str,
    batch_size: int,
    feature_set_name: str,
    loader_num_workers: int = 4,
    loader_config: Optional[data_loader.DataloaderSkipConfig] = None
) -> data_loader.SparseBatchProvider:
    if loader_config is None:
        loader_config = data_loader.DataloaderSkipConfig(
                random_fen_skipping=10,
                filtered=True, # filtering checks
        )
    # overwrite defaults
    elif loader_config.random_fen_skipping == 0 or not loader_config.filtered:
        print("[ft_perm.py] WARNING: Overwriting dataloader config to ensure some level of fen skipping and filtering, which are important for performance and correctness of ft perm finding.")
        print("[ft_perm.py]   Before overwrites: {}".format(loader_config))
        random_fen_skipping = loader_config.random_fen_skipping if loader_config.random_fen_skipping != 0 else 10
        loader_config = replace(
            loader_config,
            random_fen_skipping=random_fen_skipping,
            filtered=True,
        )
        print("[ft_perm.py]   After overwrites:  {}".format(loader_config))
    return data_loader.SparseBatchProvider(
        feature_set=feature_set_name,
        filenames=[data_path],
        cyclic=True,
        num_workers=loader_num_workers,  # some speedup and avoids StopIteration from fetch_next_fen_batch.
        batch_size=batch_size,
        config=loader_config,
    )

def quantize_ft(model: NNUEModel) -> None:
    for f in model.input.features:
        f.weight.data = f.weight.data.mul(model.quantization.ft_quantized_one).round()
        f.weight.data = f.weight.data.div_(model.quantization.ft_quantized_one)
    model.input.bias.data = model.input.bias.data.mul(model.quantization.ft_quantized_one).round()
    model.input.bias.data.div_(model.quantization.ft_quantized_one)


def eval_ft(model: NNUEModel, batch: data_loader.SparseBatchPtr, device_str: str) -> torch.Tensor:
    with torch.no_grad():
        batch_tuple = tuple(
            batch_part.to(device=device_str) for batch_part in batch
        )
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
        ) = batch_tuple
        l0_, wpsqt, bpsqt = model.forward_ft(
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            fake_quantize_acts=True,
        )
        _, _ = wpsqt, bpsqt
        return l0_


def ft_permute_impl(model: NNUEModel, perm: npt.NDArray[np.int_]) -> None:
    permutation = list(perm)

    l1_size = model.layer_stacks.l1.linear.in_features
    if l1_size != len(permutation) * 2:
        raise Exception(
            f"Invalid permutation size. Expected {l1_size}. Got {len(permutation) * 2}."
        )

    # Both sides of the FT must use the same permutation.
    permutation.extend([x + l1_size // 2 for x in permutation])

    # Add identity permutation for PSQT weights
    ft_permutation = permutation + list(range(l1_size, model.input.num_outputs))

    # Apply the permutation in place.
    for f in model.input.features:
        f.weight.data = f.weight.data[:, ft_permutation]
    model.input.bias.data = model.input.bias.data[ft_permutation]
    model.layer_stacks.l1.linear.weight.data = model.layer_stacks.l1.linear.weight.data[
        :, permutation
    ]


def ft_permute(model: NNUEModel, ft_perm_path: str) -> None:
    with open(ft_perm_path, "rb") as f:
        permutation = np.load(f)

    ft_permute_impl(model, permutation)


def gather_impl(
        model: NNUEModel,
        dataset: str,
        count: int,
        device_str: str,
        loader_workers: int = 4,
        loader_config: Optional[data_loader.DataloaderSkipConfig] = None
    ) -> npt.NDArray[np.bool_]:
    ZERO_POINT = 0.0  # Vary this to check hypothetical forced larger truncation to zero
    BATCH_SIZE = 1024

    weight_quantized_model = copy.deepcopy(model).to(device_str)
    quantize_ft(weight_quantized_model)

    sparse_batch_provider = make_sparse_batch_provider(
        dataset,
        BATCH_SIZE,
        weight_quantized_model.input_feature_name,
        loader_num_workers=loader_workers,
        loader_config=loader_config
    )

    actmats = []

    done = 0
    print("Processed {} positions.".format(done))
    while done < count:
        # checks are already filtered by sparse_batch_provider.
        s_batch = next(sparse_batch_provider)

        actmat = eval_ft(weight_quantized_model, s_batch, device_str).cpu()
        actmat = actmat <= ZERO_POINT
        actmat = actmat.numpy()
        actmats.append(actmat)

        done += len(actmat)
        print("Processed {} positions.".format(done))

    return np.concatenate(actmats, axis=0)


def command_gather(args: FeaturePermutationConfig) -> None:
    assert isinstance(args.subcommand, GatherConfig)
    if args.subcommand.checkpoint:
        nnue = NNUE.load_from_checkpoint(
            args.subcommand.checkpoint,
            feature_name=args.subcommand.feature_config.features,
            config=M.NNUELightningConfig(
                model_config=args.model_config,
            ),
            map_location=torch.device("cpu"),
        )
        model = nnue.model
    else:
        assert args.subcommand.net is not None
        model = read_model(
            args.subcommand.net,
            args.subcommand.feature_config.features,
            args.model_config,
        )

    model.eval()

    device_str = resolve_device(args.use_cupy, args.device)
    actmat = gather_impl(
        model,
        args.subcommand.data,
        args.subcommand.count,
        device_str,
        args.subcommand.loader_num_workers,
        args.subcommand.loader_config
    )

    with open(args.subcommand.out, "wb") as file:  # was: args.out
        np.save(file, actmat)


def eval_act_mat(actmat: npt.NDArray[np.bool_]) -> float:
    actmat = actmat.reshape((actmat.shape[0], actmat.shape[1] // 4, 4))
    r = np.all(actmat, axis=2)
    return np.count_nonzero(r) / r.shape[0] / r.shape[1]


def eval_perm_impl(
    actmat: npt.NDArray[np.bool_], perm: npt.NDArray[np.int_] | None = None
) -> None:
    actmat = np.reshape(actmat, (actmat.shape[0] * 2, actmat.shape[1] // 2))

    actmat_eval = eval_act_mat(actmat)
    print(f"Combined zeros in base matrix: {actmat_eval * 100:0.6f}")

    if perm is not None:
        perm_act_mat = actmat[:, perm]
        perm_act_mat_eval = eval_act_mat(perm_act_mat)
        print(f"Combined zeros in perm matrix: {perm_act_mat_eval * 100:0.6f}")


def command_eval_perm(args: FeaturePermutationConfig) -> None:
    assert isinstance(args.subcommand, EvalPermConfig)
    with open(args.subcommand.data, "rb") as file:
        actmat = np.load(file)

    if args.subcommand.perm is not None:
        with open(args.subcommand.perm, "rb") as file:
            perm = np.load(file)
    else:
        perm = None

    eval_perm_impl(actmat, perm)


def command_find_perm(args: FeaturePermutationConfig) -> None:
    assert isinstance(args.subcommand, FindPermConfig)
    with open(args.subcommand.data, "rb") as file:
        actmat = np.load(file)

    device_str = resolve_device(args.use_cupy, args.device)
    perm = find_perm_impl(actmat, device_str, args.model_config.L1)

    # perm = np.random.permutation([i for i in range(L1)])
    with open(args.subcommand.out, "wb") as file:
        np.save(file, perm)


def ft_optimize(
    model: NNUEModel,
    dataset_path: str,
    count: int,
    actmat_save_path: str | None = None,
    perm_save_path: str | None = None,
    use_cupy: bool = True,
    device: Union[int, Literal["cpu", "mps"]] = 0,
    loader_num_workers: int = 4,
    loader_config: Optional[data_loader.DataloaderSkipConfig] = None,
) -> None:
    device_str = resolve_device(use_cupy, device)

    print("Gathering activation data...")
    actmat = gather_impl(model, dataset_path, count, device_str, loader_num_workers, loader_config)
    if actmat_save_path is not None:
        with open(actmat_save_path, "wb") as file:
            np.save(file, actmat)

    print("Finding permutation...")
    perm = find_perm_impl(actmat, device_str, model.L1)
    if perm_save_path is not None:
        with open(perm_save_path, "wb") as file:
            np.save(file, perm)

    print("Evaluating permutation...")
    eval_perm_impl(actmat, perm)

    print("Applying permutation...")
    ft_permute_impl(model, perm)


def set_cupy_device(device: Optional[int]=None) -> None:
    # kept for legacy reasons.
    global _DEVICE_OVERRIDE
    _DEVICE_OVERRIDE = device

def main() -> None:
    cfg = tyro.cli(FeaturePermutationConfig)

    match cfg.subcommand:
        case GatherConfig():
            command_gather(cfg)
        case FindPermConfig():
            command_find_perm(cfg)
        case EvalPermConfig():
            command_eval_perm(cfg)


if __name__ == "__main__":
    main()
