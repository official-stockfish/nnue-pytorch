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
        model = torch.load(filename, weights_only=False)
        model.eval()
        return model.model

    elif filename.endswith(".ckpt"):
        from ..nnue import NNUE

        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
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
