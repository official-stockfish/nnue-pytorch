from collections import OrderedDict
from feature_block import *

def _calculate_features_hash(features):
    if len(features) == 1:
        return features[0].hash

    tail_hash = calculate_features_hash(features[1:])
    return features[0].hash ^ (tail_hash << 1) ^ (tail_hash >> 1) & 0xffffffff

class FeatureSet:
    def __init__(self, features):
        for feature in features:
            if not isinstance(feature, FeatureBlock):
                raise Exception('All features must subclass FeatureBlock')

        self.features = features
        self.hash = _calculate_features_hash(features)
        self.name = '+'.join(feature.name for feature in features)
        self.num_real_features = sum(feature.num_real_features for feature in features)
        self.num_virtual_features = sum(feature.num_virtual_features for feature in features)
        self.num_features = sum(feature.num_features for feature in features)

    def get_virtual_feature_ranges(self):
        ranges = []
        offset = 0
        for feature in self.features:
            if feature.num_virtual_features:
                ranges.append((offset + feature.num_real_features, offset + feature.num_features))
            offset += feature.num_features

        return ranges

    def get_active_features(self, board):
        w = []
        b = []

        offset = 0
        for feature in self.features:
            w_local, b_local = feature.get_active_features(board)
            w += [i + offset for i in w_local]
            b += [i + offset for i in b_local]
            offset += feature.num_features

        return w, b

    def get_feature_factors(self, idx):
        offset = 0
        for feature in self.features:
            if idx < offset + feature.num_real_features:
                return [offset + i for i in feature.get_feature_factors(idx - offset)]
            offset += feature.num_features

        raise Exception('No feature block to factorize {}'.format(idx))

    def get_virtual_to_real_features_gather_indices(self):
        indices = []
        real_offset = 0
        offset = 0
        for feature in self.features:
            for i_real in range(feature.num_real_features):
                i_fact = feature.get_feature_factors(i_real)
                indices.append([offset + i for i in i_fact])
            real_offset += feature.num_real_features
            offset += feature.num_features
        return indices