from dataclasses import dataclass
from typing import Annotated

import tyro


# 3 layer fully connected network
@dataclass
class ModelConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 31
    """Size of second hidden layer."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Size of third hidden layer."""

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


# parameters needed for the definition of the loss
@dataclass
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
    lambda_: float = 1.0
    """1.0=train on evaluations, 0.0=train on game results, interpolates between (default=1.0)."""

    @staticmethod
    def get_loss_params_from_args(args) -> "LossParams":
        params = LossParams()
        params.in_offset = args.in_offset
        params.out_offset = args.out_offset
        params.in_scaling = args.in_scaling
        params.out_scaling = args.out_scaling
        params.start_lambda = args.start_lambda or args.lambda_
        params.end_lambda = args.end_lambda or args.lambda_
        params.pow_exp = args.pow_exp
        params.qp_asymmetry = args.qp_asymmetry
        params.w1 = args.w1
        params.w2 = args.w2
        return params