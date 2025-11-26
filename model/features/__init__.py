import argparse
import types
from typing import Callable

from .feature_block import FeatureBlock
from .feature_set import FeatureSet

"""
Each module that defines feature blocks must be imported here and
added to the _feature_modules list. Each such module must define a
function `get_feature_block_clss` at module scope that returns the list
of feature block classes in that module.
"""
from . import halfkp, halfka, halfka_v2, halfka_v2_hm, full_threats

_feature_modules: list[types.ModuleType] = [halfkp, halfka, halfka_v2, halfka_v2_hm, full_threats]

_feature_blocks_by_name: dict[str, FeatureBlock] = dict()


def _add_feature_block(feature_block_cls: Callable[[], FeatureBlock]) -> None:
    feature_block = feature_block_cls()
    _feature_blocks_by_name[feature_block.name] = feature_block


def _add_features_blocks_from_module(module: types.ModuleType) -> None:
    feature_block_clss = module.get_feature_block_clss()
    for feature_block_cls in feature_block_clss:
        _add_feature_block(feature_block_cls)


def get_feature_block_from_name(name: str) -> FeatureBlock:
    return _feature_blocks_by_name[name]


def get_feature_blocks_from_names(names: list[str]) -> list[FeatureBlock]:
    return [_feature_blocks_by_name[name] for name in names]


def get_feature_set_from_name(name: str) -> FeatureSet:
    feature_block_names = name.split("+")
    blocks = get_feature_blocks_from_names(feature_block_names)
    return FeatureSet(blocks)


def get_available_feature_blocks_names() -> list[str]:
    return list(iter(_feature_blocks_by_name))


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    _default_feature_set_name = "HalfKAv2_hm^"
    parser.add_argument(
        "--features",
        dest="features",
        default=_default_feature_set_name,
        help='The feature set to use. Can be a union of feature blocks (for example P+HalfKP). "^" denotes a factorized block. Currently available feature blocks are: '
        + ", ".join(get_available_feature_blocks_names()),
    )


def _init() -> None:
    for module in _feature_modules:
        _add_features_blocks_from_module(module)


_init()


__all__ = [
    "FeatureSet",
    "add_feature_args",
    "get_available_feature_blocks_names",
]
