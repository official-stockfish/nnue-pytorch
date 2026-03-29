"""Drop-in replacement for Ranger21 that uses Metal kernels for the
per-parameter update, eliminating the Python-level double loop and
~50 PyTorch ops per step.

Only implements the code path that Ranger21 takes with the settings
used by this codebase (no warmup, no warmdown, no GC, no AGC, no
softplus, no madgrad, no adabelief, weight_decay=0, PNM factor=0).
"""

import math

import torch

from ..metal_support import metal_fused_adam_step_multi


class FusedRanger21(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1e-7,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

        beta1, beta2 = betas
        self._beta1_sq = beta1 ** 2
        self._one_minus_beta1_sq = 1.0 - self._beta1_sq
        self._beta2 = beta2
        self._one_minus_beta2 = 1.0 - beta2
        self._noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            params_list = []
            grads_list = []
            grad_ma_list = []
            variance_ma_list = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["grad_ma"] = torch.zeros_like(p.data)
                    state["variance_ma"] = torch.zeros_like(p.data)

                state["step"] += 1

                params_list.append(p.data)
                grads_list.append(p.grad.data)
                grad_ma_list.append(state["grad_ma"])
                variance_ma_list.append(state["variance_ma"])

            if not params_list:
                continue

            step = self.state[group["params"][0]]["step"]
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            inv_sqrt_bc2 = 1.0 / math.sqrt(bias_correction2)
            step_size = lr / bias_correction1 / self._noise_norm

            metal_fused_adam_step_multi(
                params_list, grads_list, grad_ma_list, variance_ma_list,
                self._beta1_sq, self._one_minus_beta1_sq,
                self._beta2, self._one_minus_beta2,
                inv_sqrt_bc2, step_size, eps,
            )

        return loss
