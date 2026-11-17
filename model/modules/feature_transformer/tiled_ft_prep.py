"""Index-only preparation of the tiled feature-transformer backward.

For one batch it orders the perspective lists into tiles (list_order) and sorts every tile's
entries by row (prepare). It reads only the index tensors, so the loader's prefetch thread builds
it ahead of the step on its own CUDA stream (register_prep), and the training step picks it up by
the batch's index tensor (claim_prep). See tiled_ft_kernel.py for the vocabulary.
"""

from __future__ import annotations

import collections
import threading
import warnings
import weakref
from dataclasses import dataclass

import numpy as np
import torch

from ...cupy_utils import cupy_stream, shared_memory_limit
from . import tiled_ft_kernel as kernel

# "prep_registered", "prep_hit", "fallback", "global_sort_fallback"
STATS = collections.Counter()

_warned: set[str] = set()


def warn_once(key: str, message: str) -> None:
    """Warn about a slow path the first time it is taken; STATS counts every time."""
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, stacklevel=3)


@dataclass
class TiledPrep:
    """The list order of one batch and every tile's entries sorted by row."""

    n_lists: int
    n_tiles: int
    order: torch.Tensor  # (n_lists,) int32: list ids in tile order
    tile_start: torch.Tensor  # (n_tiles + 1,) int32: offset of every tile's first entry
    entries: torch.Tensor  # (E,) uint32 keys stored as int32, sorted within each tile

    def record_stream(self, stream):
        for t in (self.order, self.tile_start, self.entries):
            t.record_stream(stream)


def _warp_per_list_grid(n_lists: int) -> tuple:
    return ((n_lists * 32 + 255) // 256,)  # blocks of 256 threads


@torch.compiler.disable
def list_order(
    white: torch.Tensor,
    black: torch.Tensor,
    buckets: tuple[int, int] | None = None,
):
    """Permutation of the perspective lists (white 0..B-1, black B..2B-1; int32) that forms the tiles,
    and the number of active rows per list (int32, list order).

    Lists are sorted by a min-hash of their rows, so lists that share rows tend to share a tile.
    ``buckets`` = (first_row, rows_per_bucket) groups them by bucket first: the highest row of every
    list falls in a bucket-major feature that starts at first_row (the last feature of the set; see
    ComposedFeatureTransformer.list_buckets). Any order gives the same sums up to fp32 rounding."""
    batch_size, max_active = white.shape
    n_lists = 2 * batch_size
    dev = white.device
    first_row, rows_per_bucket = buckets if buckets is not None else (0, 0)
    keys = torch.empty(n_lists, dtype=torch.int64, device=dev)
    counts = torch.empty(n_lists, dtype=torch.int32, device=dev)
    with cupy_stream(dev):
        kernel.order_key_kernel(max_active)(
            grid=_warp_per_list_grid(n_lists),
            block=(256,),
            args=(
                white.data_ptr(),
                black.data_ptr(),
                keys.data_ptr(),
                counts.data_ptr(),
                np.int32(n_lists),
                np.int32(batch_size),
                np.int32(first_row),
                np.int32(rows_per_bucket),
            ),
        )
    return torch.sort(keys)[1].to(torch.int32), counts


def _sort_keys_per_thread(largest_sub_tile: int) -> int | None:
    """Keys per thread with which the sub-tile sort holds ``largest_sub_tile`` keys (a multiple of
    8), or None when that is more than the kernel supports."""
    needed = -(-largest_sub_tile // (8 * kernel.SORT_THREADS)) * 8
    if needed > kernel.SORT_MAX_KEYS_PER_THREAD:
        return None
    return max(needed, kernel.SORT_MIN_KEYS_PER_THREAD)


def _sort_tiles(keys, tile_start, sub_start, keys_per_thread: int, merge_bytes: int):
    """Sort every tile's keys: radix-sort its sub-tiles in place, then merge the tile's sorted
    runs by rank in ``merge_bytes`` of shared memory (keys are unique across runs)."""
    dev = keys.device
    n_sub = sub_start.numel() - 1
    merged = torch.empty_like(keys)
    with cupy_stream(dev):
        kernel.subtile_sort_kernel(keys_per_thread)(
            grid=(n_sub,),
            block=(kernel.SORT_THREADS,),
            args=(sub_start.data_ptr(), keys.data_ptr()),
        )
        kernel.merge_runs_kernel(shared_memory_limit(dev))(
            grid=(tile_start.numel() - 1,),
            block=(1024,),
            shared_mem=merge_bytes,
            args=(
                sub_start.data_ptr(),
                tile_start.data_ptr(),
                keys.data_ptr(),
                merged.data_ptr(),
                np.int32(n_sub),
            ),
        )
    return merged


def _sort_global(keys, tile_start, n_entries: int):
    """Sort every tile's keys with one global sort by (tile, key)."""
    n_tiles = tile_start.numel() - 1
    tile_of = torch.repeat_interleave(
        torch.arange(n_tiles, device=keys.device, dtype=torch.int64),
        (tile_start[1:] - tile_start[:-1]).long(),
        output_size=n_entries,
    )  # size known: no host sync
    full = (tile_of << 32) | (keys.to(torch.int64) & 0xFFFFFFFF)
    return (torch.sort(full)[0] & 0xFFFFFFFF).to(torch.int32)


@torch.compiler.disable
def prepare(
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    buckets: tuple[int, int] | None = None,
    sort: str = "tile",
) -> TiledPrep:
    """Build the tile structure for a batch (both perspectives; only the indices are read).
    ``buckets``: see list_order. ``sort="global"`` forces the fallback used when a tile's entries
    do not fit the tile-local sort (for tests)."""
    assert white_indices.dtype == torch.int32 and white_indices.is_cuda
    dev = white_indices.device
    white_indices = white_indices.contiguous()
    black_indices = black_indices.contiguous()
    batch_size, max_active = white_indices.shape
    n_lists = 2 * batch_size
    order, counts = list_order(white_indices, black_indices, buckets)
    # offsets[p]: the first entry of the list at position p of the tile order
    offsets = torch.zeros(n_lists + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(counts[order.long()], 0)

    def group_start(lists: int):
        """Entry offsets of the groups of ``lists`` consecutive lists, and the end."""
        return torch.cat([offsets[0:n_lists:lists], offsets[n_lists:]]).contiguous()

    def largest(start):
        return (start[1:] - start[:-1]).max()

    tile_start = group_start(kernel.TILE_LISTS)
    sub_start = group_start(kernel.SUBTILE_LISTS)
    n_entries, max_tile, max_sub = torch.stack(
        [offsets[-1], largest(tile_start), largest(sub_start)]
    ).tolist()  # one host sync per batch

    keys = torch.empty(max(n_entries, 1), dtype=torch.int32, device=dev)
    with cupy_stream(dev):
        kernel.tile_key_kernel(max_active)(
            grid=_warp_per_list_grid(n_lists),
            block=(256,),
            args=(
                white_indices.data_ptr(),
                black_indices.data_ptr(),
                order.data_ptr(),
                offsets.data_ptr(),
                keys.data_ptr(),
                np.int32(n_lists),
                np.int32(batch_size),
            ),
        )

    # A tile (~23k entries for the default feature set) is too much for one block's sort, so its
    # sub-tiles are sorted and then merged; the merge holds the tile in shared memory.
    keys_per_thread = _sort_keys_per_thread(max_sub)
    merge_bytes = max(max_tile, 1) * 4
    if (
        sort == "tile"
        and keys_per_thread is not None
        and merge_bytes <= shared_memory_limit(dev)
    ):
        entries = _sort_tiles(keys, tile_start, sub_start, keys_per_thread, merge_bytes)
    else:
        if sort == "tile":
            STATS["global_sort_fallback"] += 1
            warn_once(
                "global_sort",
                f"tiled FT backward: a batch does not fit the tile-local sort (largest sub-tile "
                f"{max_sub} keys, largest tile {max_tile} entries, {shared_memory_limit(dev)} B "
                "of shared memory per block); such batches use a slower global sort.",
            )
        entries = _sort_global(keys, tile_start, n_entries)
    return TiledPrep(n_lists, tile_start.numel() - 1, order, tile_start, entries)


# Preparations parked for the training step:
# data_ptr(white_indices) -> (weak reference to that tensor, preparation)
# An entry stays until the step claims it or its batch is freed, so the table holds at most one
# entry per live batch, however long the loader's queue is.
_parked: dict[int, tuple] = {}
_parked_lock = threading.Lock()


def register_prep(
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    buckets: tuple[int, int] | None = None,
) -> None:
    """Build the preparation of a batch and park it under its white index tensor for the training
    step (claim_prep). Called on the loader's prefetch thread, on its CUDA stream."""
    prep = prepare(white_indices, black_indices, buckets)
    with _parked_lock:
        for key in [key for key, (batch, _) in _parked.items() if batch() is None]:
            del _parked[key]
        _parked[white_indices.data_ptr()] = (weakref.ref(white_indices), prep)
    STATS["prep_registered"] += 1


def claim_prep(white_indices: torch.Tensor) -> TiledPrep | None:
    """Take the preparation parked for this index tensor, if there is one and the tensor it was
    built from is still alive with this shape."""
    with _parked_lock:
        entry = _parked.pop(white_indices.data_ptr(), None)
    if entry is None:
        return None
    batch, prep = entry[0](), entry[1]
    if batch is None or batch.shape != white_indices.shape:
        return None
    # built on the prefetch stream; the consumer stream already waited on the batch's ready event
    prep.record_stream(torch.cuda.current_stream(white_indices.device))
    STATS["prep_hit"] += 1
    return prep
