# Tiled feature-transformer backward

The backward of the fused double feature transformer adds, for every position and perspective,
one gradient row to the weight-gradient row of each active feature. The plain kernel does this
with one `atomicAdd` per (active feature, column). Positions in a batch share many features, so
most of those atomics hit rows that other positions are adding to at the same time.

The tiled backward groups the work so that each weight-gradient row is added to once per group
instead of once per position. On GH200 with the default feature set it was measured at +15%
trainer it/s over the kernels it replaces there.

## Vocabulary

- A **list** is the active rows of one perspective of one position. A batch of `B` positions has
  `2B` lists: white lists `0..B-1`, black lists `B..2B-1`.
- A **tile** is 256 lists that are consecutive in the *tile order*, a permutation of the lists
  chosen so that lists which share rows land in the same tile.
- An **entry** is one (row, list) pair. It is stored as the 32-bit key
  `row << 8 | index of the list within its tile`, so sorting a tile's keys groups its entries by row.

## How a batch flows

1. **Preparation** (`tiled_ft_prep.py`, index tensors only).
   - `list_order` gives every list a sort key: the bucket of its highest row, when the last feature
     of the set is bucket-major (`InputFeature.NUM_BUCKETS`, K32H3), followed by four min-hashes of
     its rows. Sorting the keys gives the tile order.
   - `prepare` writes the entry keys tile by tile and sorts each tile. A tile holds about 23k
     entries with the default feature set, too many for one block, so its four 64-list sub-tiles
     are radix-sorted (CUB) and then merged by rank in shared memory. The sort kernel is compiled
     for the densest sub-tile of the batch (24, 32 or 40 keys per thread).
   - The result is a `TiledPrep`: the tile order, each tile's first entry, and the sorted entries.
2. **Hand-off.** The preparation needs no gradients, so the loader builds it ahead of the step.
   `DevicePrefetcher` takes a `prefetch_hook(batch)` that runs on the prefetch thread and
   stream for every device batch; `NNUE.prefetch_hook()` supplies one that calls `register_prep`.
   The preparation is parked under the address of the batch's white index tensor, and
   `FusedDoubleFtFunction.forward` claims it with `claim_prep`. An entry lives until it is claimed
   or its batch is freed.
3. **Backward** (`tiled_ft_backward.py`).
   - The prologue kernel turns the output gradient and the forward's clamped values into the
     gradients of the white and black accumulators, each `(B, L1)`.
   - The scatter kernel runs one block per (tile, 64-column slice). The block loads the tile's 256
     gradient rows for its columns into shared memory, then its warps walk the tile's sorted
     entries: consecutive entries with the same row are summed in registers and flushed with one
     `atomicAdd` per column.

The CUDA sources are in `tiled_ft_kernel.py`.

## When it runs

`tiled_ft_backward.enabled(l1_size, num_inputs, device)` decides, and everything else asks it:

- CuPy on CUDA: the kernels use 32-lane warp intrinsics and CUB, which ROCm's compiler rejects;
- `l1_size` is a multiple of 64, and the feature set has at most 2**20 rows;
- a block of the GPU can opt in to 69,632 B of shared memory (sm_75, with 64 KB, cannot).

Otherwise the fused FT uses its direct scatter (`fused_ft_kernel.py`), one `atomicAdd` per
(active feature, column).

## Slow paths

Each is counted in `tiled_ft_prep.STATS` and warned about once.

| Counter | Meaning |
|---|---|
| `fallback` | No preparation was prefetched for the batch, so the backward built it, with a host sync. Expected for code that does not pass the hook to its loader; a training run should not see it. |
| `global_sort_fallback` | A sub-tile has more than 10,240 entries, or a tile does not fit the GPU's shared memory (25,344 entries on sm_86 / sm_89). The batch is sorted with one `torch.sort`, which made the preparation about 3x slower on an RTX 3060. |

A prefetch hook that raises is disabled for the rest of the run with a `RuntimeWarning`; the
backward then falls back as above.

## Numerics and tests

The tiled backward computes the same sums as the plain kernels in a different fp32 order, as the
atomics of the plain kernels already do from run to run. `tests/test_tiled_ft.py` checks the
gradients against an fp64 reference and the plain kernels, every sort path against a global sort
and against the batch's (list, row) pairs, and that a GPU with too little shared memory keeps the
plain backward. These tests need an NVIDIA GPU; `tests/test_prefetch_hook.py` covers the hook and
the hand-off table on any CUDA or ROCm device.
