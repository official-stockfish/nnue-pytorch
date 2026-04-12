from dataclasses import dataclass, field
from typing import Annotated

import tyro
from tyro.conf import OmitArgPrefixes

from .quantize import QuantizationConfig
from .optimizers import OptimizerConfig
from .modules import FeatureConfig, LayerStacksConfig


# 3 layer fully connected network
@dataclass(kw_only=True)
class ModelConfig(LayerStacksConfig):
    @staticmethod
    def add_model_args(parser):
        parser.add_argument(
            "--l1",
            dest="L1",
            type=int,
            default=ModelConfig.L1,
        )
        parser.add_argument(
            "--l2",
            dest="L2",
            type=int,
            default=ModelConfig.L2,
        )

    @staticmethod
    def get_model_config(args) -> "ModelConfig":
        config = ModelConfig()
        config.L1 = args.L1
        config.L2 = args.L2
        return config

    gumbel_tau: float = 0.2
    """Argument for router gumbel softmax."""

    num_router_features_per_side: int = 16
    """How many features per side from ft are used for router."""
    # Not ommiting prefix on purpose.
    quantize_config: QuantizationConfig = field(default_factory=QuantizationConfig)


# parameters needed for the definition of the loss
@dataclass(kw_only=True)
class LossParams:
    in_offset: float = 270
    """offset for conversion to win on input (default=270.0)"""
    out_offset: float = 270
    """offset for conversion to win on output (default=270.0)"""
    in_scaling: float = 340
    """scaling for conversion to win on input (default=340.0)"""
    out_scaling: float = 380
    """scaling for conversion to win on output (default=380.0)"""
    start_lambda: float | None = None
    """lambda to use at first epoch."""
    end_lambda: float | None = None
    """lambda to use at last epoch."""
    pow_exp: float = 2.5
    """exponent of the power law used for the mean error (default=2.5)"""
    qp_asymmetry: float = 0.0
    """Adjust loss if q (prediction) > p (reference) (default=0.0)"""
    w1: float = 0.0
    """weight boost parameter 1 (default=0.0)"""
    w2: float = 0.5
    """weight boost parameter 2 (default=0.5)"""
    moe_loss_weight: float = 0.001
    """weight of the MoE load balancing loss (default=0.001)"""
    lambda_: Annotated[float, tyro.conf.arg(name="lambda")] = 1.0
    """1.0=train on evaluations, 0.0=train on game results, interpolates between (default=1.0)."""


@dataclass(kw_only=True)
class NNUELightningConfig(FeatureConfig):
    model_config: OmitArgPrefixes[ModelConfig] = field(default_factory=ModelConfig)
    loss_params: OmitArgPrefixes[LossParams] = field(default_factory=LossParams)
    optimizer_config: OmitArgPrefixes[OptimizerConfig] = field(
        default_factory=OptimizerConfig
    )
