from dataclasses import dataclass


# 3 layer fully connected network
@dataclass
class ModelConfig:
    L1: int = 3072
    L2: int = 15
    L3: int = 32


# parameters needed for the definition of the loss
@dataclass
class LossParams:
    in_offset: float = 270
    out_offset: float = 270
    in_scaling: float = 340
    out_scaling: float = 380
    start_lambda: float = 1.0
    end_lambda: float = 1.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0
    w1: float = 0.0
    w2: float = 0.5
