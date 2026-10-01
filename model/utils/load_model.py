import pickle

import torch

from ..config import ModelConfig, NNUELightningConfig
from ..model import NNUEModel
from .serialize import NNUEReader


def load_model(
    filename: str,
    feature_name: str,
    config: ModelConfig,
) -> NNUEModel:
    if filename.endswith(".pt"):
        model = NNUEModel(feature_name, config)
        try:
            state_dict = torch.load(filename, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError, AttributeError, KeyError):
            # Legacy .pt files were serialized as full NNUE objects.
            legacy_nnue = torch.load(filename, map_location="cpu", weights_only=False)
            state_dict = legacy_nnue.model.state_dict()
        model.load_state_dict(state_dict)
        model.eval()
        return model

    elif filename.endswith(".ckpt"):
        from ..nnue import NNUE

        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        model = NNUE(
            config=NNUELightningConfig(
                model_config=config,
                features=feature_name,
            ),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model.model

    elif filename.endswith(".nnue"):
        with open(filename, "rb") as f:
            reader = NNUEReader(f, feature_name, config)
        return reader.model

    else:
        raise ValueError("Invalid filetype: " + str(filename))
