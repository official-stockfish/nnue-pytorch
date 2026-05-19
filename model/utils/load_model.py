import torch

from .serialize import NNUEReader
from ..config import ModelConfig, NNUELightningConfig
from ..model import NNUEModel

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
        from ..lightning_module import NNUE

        model = NNUE.load_from_checkpoint(
            filename,
            config=NNUELightningConfig(model_config=config),
            map_location=torch.device("cpu"),
        )
        model.eval()
        return model.model

    elif filename.endswith(".nnue"):
        with open(filename, "rb") as f:
            reader = NNUEReader(f, feature_name, config)
        return reader.model

    else:
        raise Exception("Invalid filetype: " + str(filename))
