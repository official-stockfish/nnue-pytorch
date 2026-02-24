import argparse

from .halfka_v2_hm import HalfKav2Hm
from .full_threats import FullThreats

_FEATURES: dict[str, type] = {
    "HalfKAv2_hm^": HalfKav2Hm,
    "Full_Threats^": FullThreats,
}


def get_feature_cls(name: str) -> type:
    if not name.endswith("^") and name + "^" in _FEATURES:
        import warnings

        warnings.warn(
            f"Feature name '{name}' is deprecated, use '{name}^' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        name = name + "^"
    if name not in _FEATURES:
        raise KeyError(f"Unknown feature '{name}'. Available: {', '.join(_FEATURES)}")
    return _FEATURES[name]


def get_available_features() -> list[str]:
    return list(_FEATURES.keys())


def add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--features",
        dest="features",
        default="HalfKAv2_hm^",
        help="The feature set to use. Available: "
        + ", ".join(get_available_features()),
    )


__all__ = [
    "HalfKav2Hm",
    "FullThreats",
    "get_feature_cls",
    "get_available_features",
    "add_feature_args",
]
