"""Centralized Metal kernel availability detection and re-exports.

All Metal function imports go through this module so that the try/except +
is_available() boilerplate lives in exactly one place.
"""

MPS_AVAILABLE = False

metal_fused_double_forward_l0 = None
metal_fused_composed_double_forward_l0 = None
metal_fused_loss = None
metal_sqr_crelu = None
metal_indexed_stacked_linear = None
metal_fused_adam_step_multi = None
metal_sparse_linear = None
metal_double_sparse_linear = None

try:
    from .modules.feature_transformer.metal import (
        is_available as _metal_is_available,
        metal_fused_double_forward_l0,
        metal_fused_composed_double_forward_l0,
        metal_fused_loss,
        metal_sqr_crelu,
        metal_indexed_stacked_linear,
        metal_fused_adam_step_multi,
        metal_sparse_linear,
        metal_double_sparse_linear,
    )
    MPS_AVAILABLE = _metal_is_available()
except (ImportError, ModuleNotFoundError):
    pass
