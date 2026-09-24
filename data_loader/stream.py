import ctypes
import os  # noqa: F401

from ._native import FenBatchPtr, SparseBatchPtr, c_lib
from .config import (
    CDataloaderDDPConfig,
    CDataloaderHllConfig,
    CDataloaderSkipConfig,
    DataloaderDDPConfig,
    DataloaderHllConfig,
    DataloaderSkipConfig,
)


def _get_ddp_rank_and_world_size():
    """Get DDP rank and world size from torch.distributed if available."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    print(f"DDP rank: {rank}, world size: {world_size}", flush=True)
    return rank, world_size


def _to_c_str_array(str_list):
    c_str_array = (ctypes.c_char_p * len(str_list))()
    c_str_array[:] = [s.encode("utf-8") for s in str_list]
    return c_str_array


def create_fen_batch_stream(
    concurrency,
    filenames: list[str],
    batch_size,
    cyclic,
    config: DataloaderSkipConfig,
    ddp_config: DataloaderDDPConfig = None,
    hll_config: DataloaderHllConfig | None = None,
) -> ctypes.c_void_p:
    if ddp_config is None:
        rank, world_size = _get_ddp_rank_and_world_size()
        ddp_config = DataloaderDDPConfig(rank=rank, world_size=world_size)
    if hll_config is None:
        hll_config = DataloaderHllConfig()

    return c_lib.dll.create_fen_batch_stream(
        concurrency,
        len(filenames),
        _to_c_str_array(filenames),
        batch_size,
        cyclic,
        CDataloaderSkipConfig(config),
        CDataloaderDDPConfig(ddp_config),
        CDataloaderHllConfig(hll_config),
    )


def destroy_fen_batch_stream(stream: ctypes.c_void_p):
    c_lib.dll.destroy_fen_batch_stream(stream)


def fetch_next_fen_batch(stream: ctypes.c_void_p) -> FenBatchPtr:
    return c_lib.dll.fetch_next_fen_batch(stream)


def destroy_fen_batch(fen_batch: FenBatchPtr):
    c_lib.dll.destroy_fen_batch(fen_batch)


def create_sparse_batch_stream(
    feature_set: str,
    concurrency,
    filenames: list[str],
    batch_size,
    cyclic,
    config: DataloaderSkipConfig,
    ddp_config: DataloaderDDPConfig = None,
    hll_config: DataloaderHllConfig | None = None,
) -> ctypes.c_void_p:
    if ddp_config is None:
        rank, world_size = _get_ddp_rank_and_world_size()
        ddp_config = DataloaderDDPConfig(rank=rank, world_size=world_size)
    if hll_config is None:
        hll_config = DataloaderHllConfig()

    return c_lib.dll.create_sparse_batch_stream(
        feature_set,
        concurrency,
        len(filenames),
        _to_c_str_array(filenames),
        batch_size,
        cyclic,
        CDataloaderSkipConfig(config),
        CDataloaderDDPConfig(ddp_config),
        CDataloaderHllConfig(hll_config),
    )


def destroy_sparse_batch_stream(stream: ctypes.c_void_p):
    c_lib.dll.destroy_sparse_batch_stream(stream)


def get_sparse_batch_from_fens(
    feature_set: str, fens, scores, plies, results
) -> SparseBatchPtr:
    assert len(fens) == len(scores) == len(plies) == len(results)

    def to_c_int_array(data):
        return (ctypes.c_int * len(data))(*data)

    return c_lib.dll.get_sparse_batch_from_fens(
        feature_set.encode("utf-8"),
        len(fens),
        _to_c_str_array(fens),
        to_c_int_array(scores),
        to_c_int_array(plies),
        to_c_int_array(results),
    )


def fetch_next_sparse_batch(stream: ctypes.c_void_p) -> SparseBatchPtr:
    return c_lib.dll.fetch_next_sparse_batch(stream)


def destroy_sparse_batch(batch: SparseBatchPtr):
    c_lib.dll.destroy_sparse_batch(batch)


# --- Unique position counting (HLL) ---

def get_unique_position_stats(stream: ctypes.c_void_p) -> tuple[int, int, int]:
    """Return (preskip, total, unique) where preskip is the exact count
    of all positions read (before filtering), total is the exact count
    of positions that passed filtering, and unique is the approximate
    count of distinct positions (HLL, ~0.1% SE). Race-free; may be
    called while the stream is producing batches."""
    preskip = ctypes.c_uint64(0)
    total = ctypes.c_uint64(0)
    unique = ctypes.c_uint64(0)
    c_lib.dll.get_unique_position_stats(stream, ctypes.byref(preskip), ctypes.byref(total), ctypes.byref(unique))
    return preskip.value, total.value, unique.value


def get_hll_state_size(stream: ctypes.c_void_p) -> int:
    """Return the number of bytes needed to serialize the HLL state."""
    return c_lib.dll.get_hll_state_size(stream)


def get_hll_state(stream: ctypes.c_void_p) -> bytes:
    """Serialize the current HLL state for checkpoint storage."""
    size = get_hll_state_size(stream)
    if size == 0:
        return b""
    buf = (ctypes.c_uint8 * size)()
    written = c_lib.dll.get_hll_state(stream, buf, size)
    if written == 0:
        return b""
    return bytes(buf[:written])


def hll_count_from_state(data: bytes) -> int:
    """Compute the approximate unique count from serialized HLL state.
    Used for DDP cross-rank merge: after all_reduce MAX of registers,
    this function computes the global count from the merged state."""
    if not data:
        return 0
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    count = ctypes.c_uint64(0)
    c_lib.dll.hll_count_from_state(buf, len(data), ctypes.byref(count))
    return count.value
