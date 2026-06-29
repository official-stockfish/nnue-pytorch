import argparse
from collections.abc import Callable
from dataclasses import dataclass

from .composed import ComposedFeatureTransformer
from .full_threats import FullThreats
from .halfka_v2_hm import HalfKav2Hm
from .k16q2 import K16Q2
from .input_feature import InputFeature

import tyro
from typing import Annotated


_FEATURE_COMPONENTS: dict[str, type[InputFeature]] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "K16Q2^": K16Q2,
    "Full_Threats": FullThreats,
}


def get_feature_cls(name: str) -> list[Callable[[int], InputFeature]]:
    parts = name.split("+")
    return [_FEATURE_COMPONENTS[p] for p in parts]


def get_available_features() -> list[str]:
    return list(_FEATURE_COMPONENTS.keys())


@dataclass(kw_only=True)
class FeatureConfig:
    features: Annotated[
        str,
        tyro.conf.arg(
            help="The feature set to use. Available: "
            + ", ".join(get_available_features())
            + ". Combine with +, e.g. Full_Threats+K16Q2^"
        ),
    ] = "Full_Threats+K16Q2^"


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        dest="features",
        default="Full_Threats+K16Q2^",
        help="The feature set to use. Available: "
        + ", ".join(get_available_features())
        + ". Combine with +, e.g. Full_Threats+K16Q2^",
    )


__all__ = [
    "ComposedFeatureTransformer",
    "HalfKav2Hm",
    "K16Q2",
    "FullThreats",
    "InputFeature",
    "get_feature_cls",
    "get_available_features",
    "add_feature_args",
    "FeatureConfig",
]
