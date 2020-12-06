from collections import OrderedDict

def _get_main_factor_name(full_name):
    return full_name.replace('^', '')

class FeatureBlock:
    def __init__(self, name, factors):
        if not isinstance(factors, OrderedDict):
            raise Exception('Factors must be an collections.OrderedDict')

        self.name = name
        self.factors = factors
        self.num_real_features = factors[_get_main_factor_name(name)]
        self.num_features = sum(v for n, v in factors.items())
        self.num_virtual_features = self.num_features - self.num_real_features

    def get_main_factor_name(self):
        return _get_main_factor_name(self.name)

    def get_feature_factors(self, idx):
        return [idx]

    def get_factor_base_feature(self, name):
        offset = 0
        for n, s in self.factors.items():
            if n == name:
                return offset

            offset += s

        raise Exception('No factor named {} in {}'.format(name, self.name))
