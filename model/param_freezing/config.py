from enum import Enum
from dataclasses import dataclass
from tyro.conf import EnumChoicesFromValues

class FreezeMode(Enum):
    """Modes for parameter freezing during training."""
    FULL_TRAINING = "full-training"
    PSQT_ONLY = "psqt-only"
    FROZEN_PSQT = "frozen-psqt"
    FROZEN_PSQT_FT = "frozen-psqt-ft"

@dataclass
class ParamFreezerConfig:
    param_freeze_mode: EnumChoicesFromValues[FreezeMode] = FreezeMode.FULL_TRAINING
    """Specifies which parameters are frozen.
    - full training: No parameters frozen.
    - psqt only: Train psqt seperately as linear model.
    - frozen psqt: psqt layers are frozen, others trainable.
    - frozen psqt+ft: psqt and feature-transformer layers are frozen."""
