import torch

from .fused_ft_functions import _HAS_CUPY_KERNELS, FusedDoubleFtFunction
from .sparse_linear_functions import SparseLinearFunction


def double_feature_transform(
    us: torch.Tensor,
    them: torch.Tensor,
    white_indices: torch.Tensor,
    black_indices: torch.Tensor,
    psqt_indices: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    max_ft_activation: float,
    l1_size: int,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Resolve backend
    cupy_available = _HAS_CUPY_KERNELS
    all_cuda = (
        us.is_cuda
        and them.is_cuda
        and white_indices.is_cuda
        and black_indices.is_cuda
        and psqt_indices.is_cuda
        and weight.is_cuda
        and bias.is_cuda
    )
    cuda_capable = cupy_available and all_cuda

    if backend == "auto":
        impl = "fused" if cuda_capable else "torch"
    else:
        impl = backend

    if impl == "fused":
        if not cupy_available:
            raise RuntimeError("Fused double FT backend requested, but CuPy kernels are not available.")
        if not all_cuda:
            raise RuntimeError("Fused double FT backend requested, but not all tensors/parameters are on CUDA.")
        return FusedDoubleFtFunction.apply(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            weight,
            bias,
            max_ft_activation,
            l1_size,
        )
    elif impl in ("sparse", "torch"):
        if impl == "sparse":
            if not cupy_available:
                raise RuntimeError("Sparse backend requested, but CuPy kernels are not available.")
            if not all_cuda:
                raise RuntimeError("Sparse backend requested, but not all tensors/parameters are on CUDA.")

        assert l1_size % 2 == 0

        wp = SparseLinearFunction.apply(white_indices, weight, bias, backend=impl)
        bp = SparseLinearFunction.apply(black_indices, weight, bias, backend=impl)

        w, wpsqt = torch.split(wp, l1_size, dim=1)
        b, bpsqt = torch.split(bp, l1_size, dim=1)

        psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
        wpsqt = wpsqt.gather(1, psqt_indices_unsq)
        bpsqt = bpsqt.gather(1, psqt_indices_unsq)

        l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        # do not fake quantize sum of (quantized) weights
        l0_ = torch.clamp(l0_, 0.0, max_ft_activation)

        l0_s = torch.split(l0_, l1_size // 2, dim=1)
        l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
        l0_ = torch.cat(l0_s1, dim=1)

        return l0_, wpsqt, bpsqt
    else:
        raise ValueError(f"Invalid double FT implementation mode: {backend}")
