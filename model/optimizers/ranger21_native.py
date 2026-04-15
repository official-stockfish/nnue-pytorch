"""Pure-PyTorch replacement for the ``ranger21`` pip package.

Implements exactly the Ranger21 update rule when every optional feature is
disabled (no warmup, no warmdown, no GC, no AGC, no softplus, no madgrad,
no adabelief, weight_decay=0, PNM factor=0).  Under those settings the
per-element update reduces to:

    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_t = beta1^2 * m_{t-1} + (1 - beta1^2) * g_t
    denom = sqrt(v_t) / sqrt(1 - beta2^t) + eps
    p_t = p_{t-1} - lr / (1 - beta1^t) / noise_norm * m_t / denom

where noise_norm = sqrt((1 + beta2)^2 + beta2^2).

NOTE: the first-moment EMA uses beta1 **squared** as its coefficient, but
the bias correction still uses ``1 - beta1^t`` (not ``1 - beta1^{2t}``).
This is a quirk inherited from Ranger21's PNM formulation and is preserved
here for exact numerical equivalence.

Uses ``torch._foreach_*`` ops so the update is fused into a handful of
multi-tensor CUDA kernels instead of one Python op per parameter.
"""

import math

import torch


class Ranger21Native(torch.optim.Optimizer):
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

            # v = beta2 * v + (1 - beta2) * g^2
            torch._foreach_mul_(variance_ma_list, self._beta2)
            torch._foreach_addcmul_(
                variance_ma_list, grads_list, grads_list,
                value=self._one_minus_beta2,
            )

            # m = beta1^2 * m + (1 - beta1^2) * g
            torch._foreach_mul_(grad_ma_list, self._beta1_sq)
            torch._foreach_add_(
                grad_ma_list, grads_list, alpha=self._one_minus_beta1_sq,
            )

            # denom = sqrt(v) / sqrt(1 - beta2^step) + eps
            denoms = torch._foreach_sqrt(variance_ma_list)
            torch._foreach_mul_(denoms, inv_sqrt_bc2)
            torch._foreach_add_(denoms, eps)

            # p -= step_size * m / denom
            torch._foreach_addcdiv_(
                params_list, grad_ma_list, denoms, value=-step_size,
            )

        return loss
