import torch

from .serialize import NNUEReader
from ..config import ModelConfig
from ..features import FeatureSet
from ..model import NNUEModel
from ..quantize import QuantizationConfig


def load_model(
    filename: str,
    feature_set: FeatureSet,
    config: ModelConfig,
    quantize_config: QuantizationConfig,
) -> NNUEModel:
    if filename.endswith(".pt"):
        model = torch.load(filename, weights_only=False)
        model.eval()
        return model.model

    elif filename.endswith(".ckpt"):
        from ..lightning_module import NNUE

        model = NNUE.load_from_checkpoint(
            filename,
            feature_set=feature_set,
            config=config,
            quantize_config=quantize_config,
        )
        model.eval()
        return model.model

    elif filename.endswith(".nnue"):
        with open(filename, "rb") as f:
            reader = NNUEReader(f, feature_set, config, quantize_config)
        return reader.model

    else:
        raise Exception("Invalid filetype: " + str(filename))
