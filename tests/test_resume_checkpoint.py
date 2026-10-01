import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import NNUE
from model.config import NNUELightningConfig


def _same_weights(model_a, model_b):
    """Return True if two NNUEModels have identical state_dict tensors."""
    for (ka, pa), (kb, pb) in zip(
        model_a.state_dict().items(),
        model_b.state_dict().items(),
    ):
        if ka != kb or not torch.equal(pa, pb):
            return False
    return True


def test_resume_from_model_uses_new_hyperparameters(tmp_path):
    """Resuming from a .pt file must restore weights but use the new config."""
    old_config = NNUELightningConfig()
    old_config.loss_params.lambda_config.lambda_ = 0.9

    original = NNUE(config=old_config)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(original.model.state_dict(), checkpoint_path)

    new_config = NNUELightningConfig()
    new_config.loss_params.lambda_config.lambda_ = 0.25

    resumed = NNUE(config=new_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert isinstance(state_dict, dict)
    resumed.model.load_state_dict(state_dict)

    assert resumed.config.loss_params.lambda_config.lambda_ == 0.25
    assert _same_weights(original.model, resumed.model)


def test_resume_from_legacy_full_object_pt(tmp_path):
    """Legacy .pt files that contain a full NNUE object still load weights only."""
    old_config = NNUELightningConfig()
    old_config.loss_params.lambda_config.lambda_ = 0.8

    original = NNUE(config=old_config)
    checkpoint_path = tmp_path / "legacy_model.pt"
    # Simulate the old serialization format.
    torch.save(original, checkpoint_path)

    new_config = NNUELightningConfig()
    new_config.loss_params.lambda_config.lambda_ = 0.15

    resumed = NNUE(config=new_config)
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError, KeyError):
        legacy_nnue = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict = legacy_nnue.model.state_dict()
    resumed.model.load_state_dict(state_dict)

    assert resumed.config.loss_params.lambda_config.lambda_ == 0.15
    assert _same_weights(original.model, resumed.model)


def test_load_model_from_pt_uses_fresh_config(tmp_path):
    """model.utils.load_model must rebuild the model with the supplied config."""
    from model.utils import load_model

    config = NNUELightningConfig()
    config.loss_params.lambda_config.lambda_ = 0.7

    nnue = NNUE(config=config)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(nnue.model.state_dict(), checkpoint_path)

    loaded = load_model(
        str(checkpoint_path),
        feature_name=config.features,
        config=config.model_config,
    )
    assert _same_weights(nnue.model, loaded)
