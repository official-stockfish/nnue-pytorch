import argparse

from .callbacks import WeightClippingCallback
from .lightning_module import NNUE
from .model import NNUEModel
from .utils import coalesce_ft_weights


class SetNetworkSize(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        from . import config

        config.L1 = int(values)

from .config import L1, L2, L3, LossParams

def add_argparse_args(parser):
    parser.add_argument("--l1", type=int, default=L1, action=SetNetworkSize)


__all__ = [
    "WeightClippingCallback",
    "L1",
    "L2",
    "L3",
    "LossParams",
    "NNUE",
    "NNUEModel",
    "coalesce_ft_weights",
]
