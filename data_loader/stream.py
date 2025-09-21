import ctypes

from ._native import c_lib
from .config import CDataloaderSkipConfig, DataloaderSkipConfig
from features.feature_set import FeatureSet


def _to_c_str_array(str_list):
    c_str_array = (ctypes.c_char_p * len(str_list))()
    c_str_array[:] = [s.encode("utf-8") for s in str_list]
    return c_str_array


def create_fen_batch_stream(
    concurrency,
    filenames,
    batch_size,
    cyclic,
    config: DataloaderSkipConfig,
):
    return c_lib.dll.create_fen_batch_stream(
        concurrency,
        len(filenames),
        _to_c_str_array(filenames),
        batch_size,
        cyclic,
        CDataloaderSkipConfig(config),
    )


def destroy_fen_batch_stream(stream):
    c_lib.dll.destroy_fen_batch_stream(stream)


def fetch_next_fen_batch(stream):
    return c_lib.dll.fetch_next_fen_batch(stream)


def destroy_fen_batch(fen_batch):
    c_lib.dll.destroy_fen_batch(fen_batch)


def create_sparse_batch_stream(
    feature_set: str,
    concurrency,
    filenames,
    batch_size,
    cyclic,
    config: DataloaderSkipConfig,
):
    return c_lib.dll.create_sparse_batch_stream(
        feature_set,
        concurrency,
        len(filenames),
        _to_c_str_array(filenames),
        batch_size,
        cyclic,
        CDataloaderSkipConfig(config),
    )


def destroy_sparse_batch_stream(stream):
    c_lib.dll.destroy_sparse_batch_stream(stream)


def get_sparse_batch_from_fens(feature_set: FeatureSet, fens, scores, plies, results):
    assert len(fens) == len(scores) == len(plies) == len(results)

    def to_c_int_array(data):
        return (ctypes.c_int * len(data))(*data)

    return c_lib.dll.get_sparse_batch_from_fens(
        feature_set.name.encode("utf-8"),
        len(fens),
        _to_c_str_array(fens),
        to_c_int_array(scores),
        to_c_int_array(plies),
        to_c_int_array(results),
    )


def fetch_next_sparse_batch(stream):
    return c_lib.dll.fetch_next_sparse_batch(stream)


def destroy_sparse_batch(batch):
    c_lib.dll.destroy_sparse_batch(batch)
