import torch

from ..features import FeatureSet
from ..modules import BaseFeatureTransformer


def coalesce_ft_weights(
    feature_set: FeatureSet, layer: BaseFeatureTransformer
) -> torch.Tensor:
    weight = layer.weight.data
    indices = feature_set.get_virtual_to_real_features_gather_indices()
    weight_coalesced = weight.new_zeros(
        (feature_set.num_real_features, weight.shape[1])
    )
    for i_real, is_virtual in enumerate(indices):
        weight_coalesced[i_real, :] = sum(
            weight[i_virtual, :] for i_virtual in is_virtual
        )
    return weight_coalesced


def coalesce_ft_weights_inplace(
    feature_set: FeatureSet, layer: BaseFeatureTransformer
) -> None:
    weight = layer.weight.data
    indices = feature_set.get_virtual_to_real_features_gather_indices()
    weight_coalesced = torch.zeros_like(weight)
    for i_real, is_virtual in enumerate(indices):
        weight_coalesced[i_real, :] = sum(
            weight[i_virtual, :] for i_virtual in is_virtual
        )
    layer.weight.data = weight_coalesced
