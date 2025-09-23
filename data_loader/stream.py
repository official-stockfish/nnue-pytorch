import ctypes
import os

from ._native import c_lib, SparseBatchPtr, FenBatchPtr
from .config import (
    CDataloaderSkipConfig,
    DataloaderSkipConfig,
    CDataloaderDDPConfig,
    DataloaderDDPConfig,
)


def _get_ddp_rank_and_world_size():
    """Get DDP rank and world size from environment variables."""
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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
) -> ctypes.c_void_p:
    if ddp_config is None:
        rank, world_size = _get_ddp_rank_and_world_size()
        ddp_config = DataloaderDDPConfig(rank=rank, world_size=world_size)

    return c_lib.dll.create_fen_batch_stream(
        concurrency,
        len(filenames),
        _to_c_str_array(filenames),
        batch_size,
        cyclic,
        CDataloaderSkipConfig(config),
        CDataloaderDDPConfig(ddp_config),
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
) -> ctypes.c_void_p:
    if ddp_config is None:
        rank, world_size = _get_ddp_rank_and_world_size()
        ddp_config = DataloaderDDPConfig(rank=rank, world_size=world_size)

    return c_lib.dll.create_sparse_batch_stream(
        feature_set,
        concurrency,
        len(filenames),
        _to_c_str_array(filenames),
        batch_size,
        cyclic,
        CDataloaderSkipConfig(config),
        CDataloaderDDPConfig(ddp_config),
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
