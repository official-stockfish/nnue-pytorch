import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import tyro

from .full_threats import FullThreats
from .global_features import (
    GlobalAttacks,
    GlobalSupports,
    PawnFilePairs,
    PawnFrontierPairs,
)
from .halfka_v2_hm import HalfKav2Hm
from .input_feature import InputFeature
from .k32q2 import K32Q2
from .pp_3wide import PP3Wide

_FEATURE_COMPONENTS: dict[str, type[InputFeature]] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "K32Q2^": K32Q2,
    "Global_Supports": GlobalSupports,
    "Global_Attacks": GlobalAttacks,
    "Pawn_FrontierPairs": PawnFrontierPairs,
    "Pawn_FilePairs": PawnFilePairs,
    "Full_Threats": FullThreats,
    "PP_3Wide": PP3Wide,
}

DEFAULT_FEATURES = (
    "Global_Supports+Global_Attacks+Pawn_FrontierPairs+Pawn_FilePairs+K32Q2^"
)


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
            + ". Combine with +, e.g. Full_Threats+PP_3Wide+HalfKAv2_hm^"
        ),
    ] = DEFAULT_FEATURES


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        dest="features",
        default=DEFAULT_FEATURES,
        help="The feature set to use. Available: "
        + ", ".join(get_available_features())
        + ". Combine with +, e.g. Full_Threats+PP_3Wide+HalfKAv2_hm^",
    )


__all__ = [
    "K32Q2",
    "FeatureConfig",
    "FullThreats",
    "GlobalAttacks",
    "GlobalSupports",
    "HalfKav2Hm",
    "InputFeature",
    "PP3Wide",
    "PawnFilePairs",
    "PawnFrontierPairs",
    "add_feature_args",
    "get_available_features",
    "get_feature_cls",
]
