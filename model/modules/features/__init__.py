import argparse
from collections.abc import Callable
from dataclasses import dataclass

from .composed import ComposedFeatures
from .full_threats import FullThreats
from .halfka_v2_hm import HalfKav2Hm
from .input_feature import InputFeature

import tyro
from typing import Annotated


_FEATURE_COMPONENTS: dict[str, type[InputFeature]] = {
    "HalfKAv2_hm^": HalfKav2Hm,
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
            + ". Combine with +, e.g. Full_Threats+HalfKAv2_hm^"
        ),
    ] = "Full_Threats+HalfKAv2_hm^"


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        dest="features",
        default="Full_Threats+HalfKAv2_hm^",
        help="The feature set to use. Available: "
        + ", ".join(get_available_features())
        + ". Combine with +, e.g. Full_Threats+HalfKAv2_hm^",
    )


__all__ = [
    "ComposedFeatures",
    "HalfKav2Hm",
    "FullThreats",
    "InputFeature",
    "get_feature_cls",
    "get_available_features",
    "add_feature_args",
    "FeatureConfig",
]
