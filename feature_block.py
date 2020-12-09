from collections import OrderedDict

def _get_main_factor_name(full_name):
    return full_name.replace('^', '')

class FeatureBlock:
    '''
    This is the base class for all the network input features.
    All features must inherit from this class.
    It abstracts a named set of features in a way that
    allows seamless introduction of factorizers.

    For example a set of HalfKP features is a subclass of this class, and
    so is the HalfKP^ feature set (where "^" denotes that it's factorized).

    There are 3 fundamental things about a feature block which it needs for construction:
        - name - whatever, just please use ascii. Also we'd like "^" to be reserved to
                 denote that the block is a factorized version of some other block.
        - hash - a 32 bit unsigned integer, as defined in the original nodchip trainer
        - factors - the ordered list of named "festure subblocks". If there's more than one
                    it's assumed that it's a factorized feature block.

    More about factors, because it's the fundamental building block.
    A block can have just one factor, like HalfKP, but sometimes it's possible to
    factorize some features further. Ideally we don't want to have multiple
    features talking about the same thing when the net is actually used for play,
    because it's wasteful, but it's helpful during training because it makes it
    easier to generalize over similar positions. This is for example utilized by HalfKP^,
    which defines 3 factors: HalfKP, HalfK, and P.
    Factors are passed to the constructor as an OrderedDict from string to the number of dimensions.
    The first factor is the "real" factor (or "main" factor), one that is supposed to be used for play.
    The following factors (if any) are the "virtual" factors, and are only used for training.
    Based on these factors and their dimensions FeatureBlock defines 3 values:
        - num_real_features - the number of unfactorized features that the resulting net will use in play
        - num_virtual_features - the number of additional features used for learning
                                 that will be coalesced when converting to .nnue
        - num_features - the total number of features defined by the factors.
                         should num_real_features + num_virtual_features

    FeatureBlock provides default method implementations that abstract away the
    factorized/unfactorized nature of the feature block. These methods are described in
    their own docstrings.

    The only method that the superclass of FeatureBlock must define is
    get_active_features (def get_active_features(self, board: chess.Board)),
    which takes the board and returns the list of indices of the features
    that are active for this board.
    '''

    def __init__(self, name, hash, factors):
        if not isinstance(factors, OrderedDict):
            raise Exception('Factors must be an collections.OrderedDict')

        self.name = name
        self.hash = hash
        self.factors = factors
        self.num_real_features = factors[_get_main_factor_name(name)]
        self.num_features = sum(v for n, v in factors.items())
        self.num_virtual_features = self.num_features - self.num_real_features

    def get_main_factor_name(self):
        return _get_main_factor_name(self.name)

    '''
    This method represents the default factorizer. If your feature block
    has multiple factors you need to override this method to return
    a list of factors for a given feature.
    '''
    def get_feature_factors(self, idx):
        return [idx]

    '''
    This method takes a string name of a factor and returns the offset of the
    first feature in this factor when consulted with the sizes of the previous factors.
    '''
    def get_factor_base_feature(self, name):
        offset = 0
        for n, s in self.factors.items():
            if n == name:
                return offset

            offset += s

        raise Exception('No factor named {} in {}'.format(name, self.name))
