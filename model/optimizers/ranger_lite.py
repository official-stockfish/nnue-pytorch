# RangerLite Optimizer (2026)
# A refactored, unbloated, and bug-fixed derivative of Ranger21.
#
# Original Ranger21 implementation by @lessw2020:
# URL: https://github.com/lessw2020/Ranger21
#
# Modifications and Refactoring by @TonyCongqianWang

import warnings
import torch
import math
import collections

try:
    import cupy as cp
    import numpy as np

    _HAS_CUPY = True
except Exception:
    cp = None
    np = None
    _HAS_CUPY = False


class _RangerLiteFusedKernels:
    """Cached CuPy fused update kernels.

    Three variants are provided per optimizer flavour:
    - "full":    variance update + moment update + parameter update in one launch.
                 Used when no global variance reduction is required.
    - "variance": updates variance_ma for the stable-weight-decay normalization.
    - "phase2":  moment update + optional stable weight decay + parameter update,
                 reading pre-computed variance_ma.
    """

    _compiled = False

    @classmethod
    def _compile(cls):
        if cls._compiled or not _HAS_CUPY:
            return
        source = r'''
typedef long long int64_t;

extern "C" __global__ void ranger_lite_update_pnm(
    float* __restrict__ p,
    const float* __restrict__ grad,
    float* __restrict__ grad_ma,
    float* __restrict__ neg_grad_ma,
    float* __restrict__ variance_ma,
    int64_t n,
    float beta2,
    float one_minus_beta2,
    float bias_correction2,
    float beta1_sq,
    float one_minus_beta1_sq,
    float one_plus_pnm,
    float pnm_factor,
    float noise_norm,
    float lr_div_bc1,
    float eps,
    int64_t step
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = grad[i];
        float vm = variance_ma[i] * beta2 + g * g * one_minus_beta2;
        variance_ma[i] = vm;
        float denom = sqrtf(vm / bias_correction2) + eps;

        float gm = grad_ma[i];
        float ngm = neg_grad_ma[i];
        float pgm = (step & 1LL) ? gm : ngm;
        float pngm = (step & 1LL) ? ngm : gm;

        float new_pgm = pgm * beta1_sq + g * one_minus_beta1_sq;
        float pnm_val = (new_pgm * one_plus_pnm - pngm * pnm_factor) / noise_norm;
        float new_p = p[i] - lr_div_bc1 * (pnm_val / denom);
        p[i] = new_p;

        if (step & 1LL) {
            grad_ma[i] = new_pgm;
        } else {
            neg_grad_ma[i] = new_pgm;
        }
    }
}

extern "C" __global__ void ranger_lite_update_adam(
    float* __restrict__ p,
    const float* __restrict__ grad,
    float* __restrict__ grad_ma,
    float* __restrict__ variance_ma,
    int64_t n,
    float beta2,
    float one_minus_beta2,
    float bias_correction2,
    float beta1,
    float one_minus_beta1,
    float lr_div_bc1,
    float eps
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = grad[i];
        float vm = variance_ma[i] * beta2 + g * g * one_minus_beta2;
        variance_ma[i] = vm;
        float denom = sqrtf(vm / bias_correction2) + eps;

        float gm = grad_ma[i] * beta1 + g * one_minus_beta1;
        grad_ma[i] = gm;
        p[i] = p[i] - lr_div_bc1 * (gm / denom);
    }
}

extern "C" __global__ void ranger_lite_update_variance(
    const float* __restrict__ grad,
    float* __restrict__ variance_ma,
    int64_t n,
    float beta2,
    float one_minus_beta2
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = grad[i];
        variance_ma[i] = variance_ma[i] * beta2 + g * g * one_minus_beta2;
    }
}

extern "C" __global__ void ranger_lite_update_pnm_phase2(
    float* __restrict__ p,
    const float* __restrict__ grad,
    float* __restrict__ grad_ma,
    float* __restrict__ neg_grad_ma,
    const float* __restrict__ variance_ma,
    int64_t n,
    float bias_correction2,
    float beta1_sq,
    float one_minus_beta1_sq,
    float one_plus_pnm,
    float pnm_factor,
    float noise_norm,
    float lr_div_bc1,
    float eps,
    int64_t step,
    float wd_factor
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float vm = variance_ma[i];
        float denom = sqrtf(vm / bias_correction2) + eps;

        float g = grad[i];
        float gm = grad_ma[i];
        float ngm = neg_grad_ma[i];
        float pgm = (step & 1LL) ? gm : ngm;
        float pngm = (step & 1LL) ? ngm : gm;

        float new_pgm = pgm * beta1_sq + g * one_minus_beta1_sq;
        float pnm_val = (new_pgm * one_plus_pnm - pngm * pnm_factor) / noise_norm;
        float new_p = p[i];
        if (wd_factor != 0.0f) {
            new_p = new_p * (1.0f - wd_factor);
        }
        new_p = new_p - lr_div_bc1 * (pnm_val / denom);
        p[i] = new_p;

        if (step & 1LL) {
            grad_ma[i] = new_pgm;
        } else {
            neg_grad_ma[i] = new_pgm;
        }
    }
}

extern "C" __global__ void ranger_lite_update_adam_phase2(
    float* __restrict__ p,
    const float* __restrict__ grad,
    float* __restrict__ grad_ma,
    const float* __restrict__ variance_ma,
    int64_t n,
    float bias_correction2,
    float beta1,
    float one_minus_beta1,
    float lr_div_bc1,
    float eps,
    float wd_factor
) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float vm = variance_ma[i];
        float denom = sqrtf(vm / bias_correction2) + eps;

        float g = grad[i];
        float gm = grad_ma[i] * beta1 + g * one_minus_beta1;
        grad_ma[i] = gm;

        float new_p = p[i];
        if (wd_factor != 0.0f) {
            new_p = new_p * (1.0f - wd_factor);
        }
        new_p = new_p - lr_div_bc1 * (gm / denom);
        p[i] = new_p;
    }
}
'''
        cls._pnm_kernel = cp.RawKernel(source, 'ranger_lite_update_pnm')
        cls._adam_kernel = cp.RawKernel(source, 'ranger_lite_update_adam')
        cls._variance_kernel = cp.RawKernel(source, 'ranger_lite_update_variance')
        cls._pnm_phase2_kernel = cp.RawKernel(source, 'ranger_lite_update_pnm_phase2')
        cls._adam_phase2_kernel = cp.RawKernel(source, 'ranger_lite_update_adam_phase2')
        cls._pnm_kernel.compile()
        cls._adam_kernel.compile()
        cls._variance_kernel.compile()
        cls._pnm_phase2_kernel.compile()
        cls._adam_phase2_kernel.compile()
        cls._compiled = True


class RangerLite(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-7,
        pnm_activate=True,
        pnm_momentum=1.0,
        lookahead_active=True,
        lookahead_mergetime=5,
        lookahead_blending_alpha=0.5,
        normloss_active=True,
        normloss_factor=1e-4,
        use_legacy_scoping_bug=False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            pnm_momentum=pnm_momentum,
            normloss_factor=normloss_factor,
        )
        super().__init__(params, defaults)

        self.lookahead_active = lookahead_active
        self.lookahead_mergetime = lookahead_mergetime
        self.lookahead_alpha = lookahead_blending_alpha
        self.lookahead_step = 0

        self.pnm_active = pnm_activate
        self.normloss_active = normloss_active
        self.eps = eps
        self.param_size = 0

        self.use_legacy_scoping_bug = use_legacy_scoping_bug

        if not _HAS_CUPY and torch.cuda.is_available():
            warnings.warn(
                "RangerLite: CuPy is not available; using the Python update path. "
                "Install CuPy to enable the fused optimizer kernel on CUDA.",
                stacklevel=2,
            )
        elif _HAS_CUPY and (self.normloss_active or self.use_legacy_scoping_bug):
            warnings.warn(
                "RangerLite: normloss or legacy mode disables the fused CUDA "
                "update kernel; using the Python update path.",
                stacklevel=2,
            )

    def _init_state(self, p):
        """Lazy state initialization for a parameter."""
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["variance_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if self.lookahead_active:
                state["lookahead_params"] = torch.clone(p.data)
            if self.pnm_active:
                state["neg_grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def _can_fuse_group_full(self, group):
        """Full fused path requires no decay and no normloss."""
        if not _HAS_CUPY or self.use_legacy_scoping_bug or self.normloss_active:
            return False
        if group.get("weight_decay", 0.0) != 0.0:
            return False
        return self._can_fuse_group_phase2(group)

    def _can_fuse_group_phase2(self, group):
        """Phase-2 fused path allows weight decay but not normloss/legacy."""
        if not _HAS_CUPY or self.use_legacy_scoping_bug or self.normloss_active:
            return False
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                return False
            if p.dtype != torch.float32 or p.device.type != "cuda":
                return False
            if not p.is_contiguous() or not grad.is_contiguous():
                return False
            state = self.state[p]
            if len(state) == 0:
                continue
            if (
                state["grad_ma"].dtype != torch.float32
                or state["grad_ma"].device != p.device
                or not state["grad_ma"].is_contiguous()
            ):
                return False
            if (
                state["variance_ma"].dtype != torch.float32
                or state["variance_ma"].device != p.device
                or not state["variance_ma"].is_contiguous()
            ):
                return False
            if self.pnm_active and (
                state["neg_grad_ma"].dtype != torch.float32
                or state["neg_grad_ma"].device != p.device
                or not state["neg_grad_ma"].is_contiguous()
            ):
                return False
        return True

    def _fused_update_group(self, group):
        """Apply the fused full update kernel (variance + moment + param)."""
        _RangerLiteFusedKernels._compile()
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        pnm_factor = group["pnm_momentum"]
        eps = group["eps"]
        one_minus_beta2 = 1.0 - beta2
        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        one_plus_pnm = 1.0 + pnm_factor
        noise_norm = math.sqrt((1.0 + pnm_factor) ** 2 + pnm_factor ** 2)

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            self._init_state(p)

            state["step"] += 1
            step = state["step"]
            bias_correction2 = 1.0 - beta2 ** step

            p_arr = cp.from_dlpack(torch.to_dlpack(p.detach()))
            g_arr = cp.from_dlpack(torch.to_dlpack(grad))
            gm_arr = cp.from_dlpack(torch.to_dlpack(state["grad_ma"]))
            vm_arr = cp.from_dlpack(torch.to_dlpack(state["variance_ma"]))

            block = 512
            grid = (p.numel() + block - 1) // block

            if self.pnm_active:
                effective_step = ((step + 1) // 2) * 2
                bias_correction1 = 1.0 - beta1 ** effective_step
                lr_div_bc1 = lr / bias_correction1
                ngm_arr = cp.from_dlpack(torch.to_dlpack(state["neg_grad_ma"]))
                args = (
                    p_arr,
                    g_arr,
                    gm_arr,
                    ngm_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(beta2),
                    np.float32(one_minus_beta2),
                    np.float32(bias_correction2),
                    np.float32(beta1_sq),
                    np.float32(one_minus_beta1_sq),
                    np.float32(one_plus_pnm),
                    np.float32(pnm_factor),
                    np.float32(noise_norm),
                    np.float32(lr_div_bc1),
                    np.float32(eps),
                    np.int64(step),
                )
                _RangerLiteFusedKernels._pnm_kernel(grid=(grid,), block=(block,), args=args)
            else:
                bias_correction1 = 1.0 - beta1 ** step
                lr_div_bc1 = lr / bias_correction1
                args = (
                    p_arr,
                    g_arr,
                    gm_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(beta2),
                    np.float32(one_minus_beta2),
                    np.float32(bias_correction2),
                    np.float32(beta1),
                    np.float32(1.0 - beta1),
                    np.float32(lr_div_bc1),
                    np.float32(eps),
                )
                _RangerLiteFusedKernels._adam_kernel(grid=(grid,), block=(block,), args=args)

    def _fused_update_group_phase2(self, group, variance_normalized=None):
        """Apply the fused phase-2 update kernel (moment + optional decay + param).

        Assumes variance_ma has already been updated by the caller.
        """
        _RangerLiteFusedKernels._compile()
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        decay = group.get("weight_decay", 0.0)
        pnm_factor = group["pnm_momentum"]
        eps = group["eps"]
        beta1_sq = beta1 * beta1
        one_minus_beta1_sq = 1.0 - beta1_sq
        one_plus_pnm = 1.0 + pnm_factor
        noise_norm = math.sqrt((1.0 + pnm_factor) ** 2 + pnm_factor ** 2)

        wd_factor = 0.0
        if decay and variance_normalized is not None:
            wd_factor = float(decay * lr / variance_normalized.item())

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            step = state["step"]
            bias_correction2 = 1.0 - beta2 ** step

            p_arr = cp.from_dlpack(torch.to_dlpack(p.detach()))
            g_arr = cp.from_dlpack(torch.to_dlpack(grad))
            gm_arr = cp.from_dlpack(torch.to_dlpack(state["grad_ma"]))
            vm_arr = cp.from_dlpack(torch.to_dlpack(state["variance_ma"]))

            block = 512
            grid = (p.numel() + block - 1) // block

            if self.pnm_active:
                effective_step = ((step + 1) // 2) * 2
                bias_correction1 = 1.0 - beta1 ** effective_step
                lr_div_bc1 = lr / bias_correction1
                ngm_arr = cp.from_dlpack(torch.to_dlpack(state["neg_grad_ma"]))
                args = (
                    p_arr,
                    g_arr,
                    gm_arr,
                    ngm_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(bias_correction2),
                    np.float32(beta1_sq),
                    np.float32(one_minus_beta1_sq),
                    np.float32(one_plus_pnm),
                    np.float32(pnm_factor),
                    np.float32(noise_norm),
                    np.float32(lr_div_bc1),
                    np.float32(eps),
                    np.int64(step),
                    np.float32(wd_factor),
                )
                _RangerLiteFusedKernels._pnm_phase2_kernel(grid=(grid,), block=(block,), args=args)
            else:
                bias_correction1 = 1.0 - beta1 ** step
                lr_div_bc1 = lr / bias_correction1
                args = (
                    p_arr,
                    g_arr,
                    gm_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(bias_correction2),
                    np.float32(beta1),
                    np.float32(1.0 - beta1),
                    np.float32(lr_div_bc1),
                    np.float32(eps),
                    np.float32(wd_factor),
                )
                _RangerLiteFusedKernels._adam_phase2_kernel(grid=(grid,), block=(block,), args=args)

    def unit_norm(self, x):
        """
        Calculates the L2 norm of each sub-unit (row/filter) in a parameter tensor.
        Returns a tensor of norms with the same number of dimensions as x (via keepdim).
        Examples:
            - Linear: (out, in) -> Norm per neuron
            - Conv2d: (out, in, h, w) -> Norm per filter
            - Embedding: (vocab, dim) -> Norm per vector
        """
        xlen = x.ndim

        if xlen <= 1:
            return x.norm(p=2.0, keepdim=False)
        dim = tuple(range(1, xlen))
        return x.norm(dim=dim, keepdim=True, p=2.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None and isinstance(closure, collections.abc.Callable):
            with torch.enable_grad():
                loss = closure()

        first_param = next(
            (p for group in self.param_groups for p in group['params'] if p.grad is not None),
            None,
        )

        if first_param is None:
            return loss # No grads to process

        needs_variance_sum = (
            self.normloss_active
            or any(g.get("weight_decay", 0.0) != 0.0 for g in self.param_groups)
        )

        # Fast path A: no variance sum needed and all groups can use the single
        # fused kernel that updates variance, moments and parameters at once.
        if not needs_variance_sum:
            all_fusable = all(self._can_fuse_group_full(g) for g in self.param_groups)
            if all_fusable:
                for group in self.param_groups:
                    self._fused_update_group(group)
                if self.lookahead_active:
                    self.lookahead_process_step()
                return loss

        # Fast path B: every group that has work can use the phase-2 fused kernel.
        # We run a fused variance kernel per parameter, reduce the debiased sums,
        # then run the phase-2 update kernel per parameter.
        all_phase2_fusable = all(
            self._can_fuse_group_phase2(g) for g in self.param_groups
        )
        if all_phase2_fusable:
            _RangerLiteFusedKernels._compile()

            active_params = []
            param_size = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self._init_state(p)
                    state = self.state[p]
                    state["step"] += 1
                    active_params.append((p, group))
                    param_size += p.numel()

            if not active_params:
                return loss

            for p, group in active_params:
                state = self.state[p]
                beta1, beta2 = group["betas"]
                grad = p.grad
                vm_arr = cp.from_dlpack(torch.to_dlpack(state["variance_ma"]))
                g_arr = cp.from_dlpack(torch.to_dlpack(grad))

                block = 512
                grid = (p.numel() + block - 1) // block
                args = (
                    g_arr,
                    vm_arr,
                    np.int64(p.numel()),
                    np.float32(beta2),
                    np.float32(1.0 - beta2),
                )
                _RangerLiteFusedKernels._variance_kernel(grid=(grid,), block=(block,), args=args)

            variance_normalized = None
            if needs_variance_sum:
                device = active_params[0][0].device
                variance_ma_sum = None
                for p, group in active_params:
                    state = self.state[p]
                    beta1, beta2 = group["betas"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    partial = state["variance_ma"].sum() / bias_correction2
                    if variance_ma_sum is None:
                        variance_ma_sum = partial
                    else:
                        variance_ma_sum += partial
                if not self.param_size:
                    self.param_size = param_size
                variance_normalized = torch.sqrt(variance_ma_sum / self.param_size).clamp_min(self.eps)

            for group in self.param_groups:
                self._fused_update_group_phase2(group, variance_normalized)

            if self.lookahead_active:
                self.lookahead_process_step()
            return loss

        variance_ma_sum = torch.zeros(1, device=first_param.device)
        param_size = 0
        leaked_p = None

        # Phase 1: Accumulate variance_ma_sum for stable weight decay
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                leaked_p = p
                param_size += p.numel()
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Sparse matrix not supported")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["variance_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if self.lookahead_active:
                        state["lookahead_params"] = torch.clone(p.data)

                    if self.pnm_active:
                        # PNM components
                        state["neg_grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                beta1, beta2 = group["betas"]

                # Variance computation
                bias_correction2 = 1 - beta2 ** state["step"]
                variance_ma = state["variance_ma"]

                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                variance_ma_debiased = variance_ma / bias_correction2
                variance_ma_sum += variance_ma_debiased.sum()

        if not self.param_size:
            if not param_size:
                # No trainable params
                return loss
            self.param_size = param_size

        variance_normalized = torch.sqrt(variance_ma_sum / self.param_size).clamp_min(self.eps)

        # Phase 2: Apply weight decay and update weights
        for group in self.param_groups:
            decay = group["weight_decay"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            pnm_factor = group["pnm_momentum"]
            normloss_factor = group["normloss_factor"]

            # --- LEGACY BUG BEHAVIOR ---
            # This correctly targets the last parameter of the PREVIOUS group
            # (or the absolute last parameter of the network for Group 0)
            if self.use_legacy_scoping_bug and leaked_p is not None:
                if decay:
                    leaked_p.data.mul_(1 - decay * lr / variance_normalized)
                if self.normloss_active:
                    unorm = self.unit_norm(leaked_p.data)
                    corr = 2 * normloss_factor * (1 - torch.div(1, unorm + self.eps))
                    leaked_p.data.mul_(1 - lr * corr)
            # ---------------------------

            for p in group["params"]:
                # --- MIMIC PYTHON VARIABLE LEAK ---
                # Python reassigns 'p' at the start of every loop iteration.
                # We capture that reassignment so the next group targets this group's final tensor.
                if self.use_legacy_scoping_bug:
                    leaked_p = p

                if p.grad is None:
                    continue

                state = self.state[p]

                # --- CORRECT BEHAVIOR ---
                if not self.use_legacy_scoping_bug:
                    # Stable Weight Decay
                    if decay:
                        p.data.mul_(1 - decay * lr / variance_normalized)
                    # Norm Loss
                    if self.normloss_active:
                        unorm = self.unit_norm(p.data)
                        corr = 2 * normloss_factor * (1 - torch.div(1, unorm + self.eps))
                        p.data.mul_(1 - lr * corr)
                # ------------------------

                grad = p.grad
                step = state["step"]

                grad_ma = state["grad_ma"]
                variance_ma = state["variance_ma"]

                bias_correction2 = 1 - beta2 ** step

                # Despite the comment. Ranger21 doesnt actually use variance_ma_max for denominator
                denom = (variance_ma.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                if self.pnm_active:
                    # PNM Adam Core Setup
                    if step % 2 == 1:
                        pnm_grad_ma, pnm_neg_grad_ma = state["grad_ma"], state["neg_grad_ma"]
                    else:
                        pnm_grad_ma, pnm_neg_grad_ma = state["neg_grad_ma"], state["grad_ma"]

                    # Update neg_grad_ma
                    pnm_grad_ma.mul_(beta1 ** 2).add_(grad, alpha=(1 - beta1 ** 2))

                    if self.use_legacy_scoping_bug:
                        # Legacy calculation
                        bias_correction1 = 1 - beta1 ** step
                        noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)

                    else:
                        # Corrected: Bias updated with exact number of effective steps
                        effective_step = ((step + 1) // 2) * 2
                        bias_correction1 = 1 - beta1 ** effective_step
                        # Corrected: Normalization calculated from pnm_factor like in original paper. However lr tuning is likely still required.
                        noise_norm = math.sqrt((1 + pnm_factor) ** 2 + pnm_factor ** 2)

                    pnm_val = (
                        pnm_grad_ma.mul(1 + pnm_factor)
                        .add(pnm_neg_grad_ma, alpha=-pnm_factor)
                        .mul(1 / noise_norm)
                    )

                    step_size = lr / bias_correction1
                    p.addcdiv_(pnm_val, denom, value=-step_size)

                else:
                    # Standard Adam update
                    bias_correction1 = 1 - beta1 ** step

                    # Standard EMA for the first moment
                    grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)

                    step_size = lr / bias_correction1
                    p.addcdiv_(grad_ma, denom, value=-step_size)

        if self.lookahead_active:
            self.lookahead_process_step()

        return loss


    #   Lookahead merge process
    def lookahead_process_step(self):
        """handles blending of params for lookahead step"""

        if not self.lookahead_active:
            return
        self.lookahead_step += 1

        if self.lookahead_step >= self.lookahead_mergetime:
            self.lookahead_step = 0
            # merge lookahead cached params and save current ones
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]

                    p.data.mul_(self.lookahead_alpha).add_(
                        param_state["lookahead_params"],
                        alpha=1.0 - self.lookahead_alpha,
                    )
                    # save for next merge
                    param_state["lookahead_params"].copy_(p.data)

    def train(self):
        """Switches optimizer to training mode. For optimizers with train/eval behavior, this should enable training behavior."""
        if self.lookahead_active:
            self.restore_for_training()

    def eval(self):
        """Switches optimizer to eval mode. For optimizers with train/eval behavior, this should enable evaluation behavior (e.g. swapping in slow weights)."""
        if self.lookahead_active:
            self.swap_for_inference()

    def swap_for_inference(self):
        """Safely loads slow weights for eval/saving. Idempotent."""
        if not self.lookahead_active:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "lookahead_params" in state:
                    if "is_swapped" not in state:
                        # Only swap if we haven't already swapped.
                        # Perform pointer swap to avoid unnecessary tensor copies.
                        tmp_ref = p.data
                        p.data = state["lookahead_params"]
                        state["lookahead_params"] = tmp_ref
                        state["is_swapped"] = True

    def restore_for_training(self):
        """Restores fast weights for training. Idempotent."""
        if not self.lookahead_active:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "is_swapped" in state:
                    # Intentionally overwrite lookahead_params with the current
                    # to preserve any changes made during eval mode (e.g. swa)
                    tmp_ref = state["lookahead_params"]
                    state["lookahead_params"] = p.data
                    p.data = tmp_ref
                    del state["is_swapped"]
