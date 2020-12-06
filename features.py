from feature_block import *
from feature_set import *

import halfkp

_feature_blocks_by_name = dict()

def _add_feature_block(feature_block_cls):
    feature_block = feature_block_cls()
    _feature_blocks_by_name[feature_block.name] = feature_block

_add_feature_block(halfkp.Features)
_add_feature_block(halfkp.FactorizedFeatures)

def get_feature_block_from_name(name):
    return _feature_blocks_by_name[name]

def get_feature_blocks_from_names(names):
    return [_feature_blocks_by_name[name] for name in names]

def get_feature_set_from_name(name):
    feature_block_names = name.split('+')
    blocks = get_feature_blocks_from_names(feature_block_names)
    return FeatureSet(blocks)

def get_available_feature_blocks_names():
    return list(iter(_feature_blocks_by_name))
