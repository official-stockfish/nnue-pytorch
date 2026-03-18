from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Any
from tyro.conf import EnumChoicesFromValues

class FreezeMode(Enum):
    """Modes for parameter freezing during training."""
    FULL_TRAINING = "full-training"
    PSQT_ONLY = "psqt-only"
    FROZEN_PSQT = "frozen-pqst"
    FROZEN_PSQT_FT = "frozen-pqst-ft"

@dataclass
class ParamFreezerConfig:
    param_freeze_mode: EnumChoicesFromValues[FreezeMode] = FreezeMode.FULL_TRAINING
    """Specifies which parameters are frozen.
    - full training: No parameters frozen.
    - pqst only: Train pqst seperately as linear model.
    - frozen pqst: pqst layers are frozen, others trainable.
    - frozen pqst+ft: pqst and feature-transformer layers are frozen."""
