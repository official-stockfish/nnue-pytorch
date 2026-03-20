import io
import torch

from torch import nn
from typing import Union, Iterable

from ..utils.serialize import NNUEWriter, NNUEReader

from .config import FreezeMode

TorchTarget = Union[nn.Module, nn.Parameter, Iterable[Union[nn.Module, nn.Parameter]]]

class ParamFreezer:
    def __init__(self, nnue_lightning_config):
        self.nnue_lightning_config = nnue_lightning_config
        self.mode = nnue_lightning_config.freeze_config.param_freeze_mode

    def apply_freeze(self, model: nn.Module):
        """Applies freezing logic to the model based on the selected mode."""
        # unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        # never train psqt biases
        self._set_requires_grad(model.input.get_psqt_params(bias_only=True), False)
        if self.mode == FreezeMode.FULL_TRAINING:
            return

        elif self.mode == FreezeMode.PSQT_ONLY:
            print("Training only PSQT")
            self._set_requires_grad(model.input.get_ft_params(), False)
            self._set_requires_grad(model.layer_stacks, False)
            model.psqt_only = True

        elif self.mode == FreezeMode.FROZEN_PSQT:
            print("Training with Frozen PSQT")
            self._set_requires_grad(model.input.get_psqt_params(include_bias=False), False)
            model.psqt_only = False

        elif self.mode == FreezeMode.FROZEN_PSQT_FT:
            print("Training with Frozen PSQT and FT")
            self._set_requires_grad(model.input.get_psqt_params(include_bias=False), False)
            self._set_requires_grad(model.input.get_ft_params(), False)
            model.psqt_only = False

        quantized_model = self._get_quantized_copy(model)
        self._replace_frozen_weights(model, quantized_model)

    def _set_requires_grad(
        self,
        targets: TorchTarget,
        requires_grad: bool
    ) -> None:
        """
        Sets requires_grad for Modules, Parameters, or Iterables of both.
        """
        if isinstance(targets, (nn.Module, nn.Parameter)):
            targets = [targets]

        for item in targets:
            if isinstance(item, nn.Module):
                # Use the in-place Module method for efficiency
                item.requires_grad_(requires_grad)
            elif isinstance(item, nn.Parameter):
                item.requires_grad = requires_grad
            else:
                raise TypeError(f"Expected nn.Module or nn.Parameter, got {type(item)}")

    def _get_quantized_copy(self, model: nn.Module):
        writer = NNUEWriter(
            model,
            verbose=False,
        )
        buffer_stream = io.BytesIO(writer.buf)
        reader = NNUEReader(
            buffer_stream,
            feature_name=model.feature_name,
            config=self.nnue_lightning_config.model_config,
        )

        return reader.model

    def _replace_frozen_weights(self, target_model: nn.Module, src_model: nn.Module):
        restored_state = dict(src_model.named_parameters())
        with torch.no_grad():
            for name, param in target_model.named_parameters():
                if not param.requires_grad:
                    if name not in restored_state:
                        raise KeyError(f"Frozen parameter '{name}' not found in the deserialized reader model.")
                    param.copy_(restored_state[name])


