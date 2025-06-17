from feature_block import *
from feature_set import *
import model as M

import argparse

"""
Each module that defines feature blocks must be imported here and
added to the _feature_modules list. Each such module must define a
function `get_feature_block_clss` at module scope that returns the list
of feature block classes in that module.
"""
import halfkp
import halfka
import halfka_v2
import halfka_v2_hm

_feature_modules = [halfkp, halfka, halfka_v2, halfka_v2_hm]

_feature_blocks_by_name = dict()


class SetNetworkSize(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        M.L1 = int(values)


def _add_feature_block(feature_block_cls):
    feature_block = feature_block_cls()
    _feature_blocks_by_name[feature_block.name] = feature_block


def _add_features_blocks_from_module(module):
    feature_block_clss = module.get_feature_block_clss()
    for feature_block_cls in feature_block_clss:
        _add_feature_block(feature_block_cls)


def get_feature_block_from_name(name):
    return _feature_blocks_by_name[name]


def get_feature_blocks_from_names(names):
    return [_feature_blocks_by_name[name] for name in names]


def get_feature_set_from_name(name):
    feature_block_names = name.split("+")
    blocks = get_feature_blocks_from_names(feature_block_names)
    return FeatureSet(blocks)


def get_available_feature_blocks_names():
    return list(iter(_feature_blocks_by_name))


def add_argparse_args(parser):
    _default_feature_set_name = "HalfKAv2_hm^"
    parser.add_argument(
        "--features",
        dest="features",
        default=_default_feature_set_name,
        help='The feature set to use. Can be a union of feature blocks (for example P+HalfKP). "^" denotes a factorized block. Currently available feature blocks are: '
        + ", ".join(get_available_feature_blocks_names()),
    )
    parser.add_argument("--l1", type=int, default=M.L1, action=SetNetworkSize)


def _init():
    for module in _feature_modules:
        _add_features_blocks_from_module(module)


_init()
