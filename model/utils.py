from torch import nn

from .model import NNUEModel
from .feature_transformer import BaseFeatureTransformerSlice


def coalesce_ft_weights(model: NNUEModel, layer: BaseFeatureTransformerSlice):
    weight = layer.weight.data
    indices = model.feature_set.get_virtual_to_real_features_gather_indices()
    weight_coalesced = weight.new_zeros(
        (model.feature_set.num_real_features, weight.shape[1])
    )
    for i_real, is_virtual in enumerate(indices):
        weight_coalesced[i_real, :] = sum(
            weight[i_virtual, :] for i_virtual in is_virtual
        )
    return weight_coalesced


def get_parameters(layers: list[nn.Module]):
    return [p for layer in layers for p in layer.parameters()]
