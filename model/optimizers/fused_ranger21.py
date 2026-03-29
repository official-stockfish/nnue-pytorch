"""Drop-in replacement for Ranger21 that uses Metal kernels for the
per-parameter update, eliminating the Python-level double loop and
~50 PyTorch ops per step.

Only implements the code path that Ranger21 takes with the settings
used by this codebase (no warmup, no warmdown, no GC, no AGC, no
softplus, no madgrad, no adabelief, weight_decay=0, PNM factor=0).
"""

import math

import torch

from ..modules.feature_transformer.metal import metal_fused_adam_step


class FusedRanger21(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1e-7,
        num_batches_per_epoch=None,
        num_epochs=None,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_epochs = num_epochs
        self.total_iterations = num_epochs * num_batches_per_epoch
        self.starting_lr = lr
        self.current_lr = lr
        self.epoch_count = 0
        self.current_iter = 0
        self.tracking_lr = []

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

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["grad_ma"] = torch.zeros_like(p.data)
                    state["variance_ma"] = torch.zeros_like(p.data)

                state["step"] += 1
                step = state["step"]

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                inv_sqrt_bc2 = 1.0 / math.sqrt(bias_correction2)
                step_size = lr / bias_correction1 / self._noise_norm

                metal_fused_adam_step(
                    p, grad,
                    state["grad_ma"], state["variance_ma"],
                    self._beta1_sq, self._one_minus_beta1_sq,
                    self._beta2, self._one_minus_beta2,
                    inv_sqrt_bc2, step_size, eps,
                )

        self._track_epochs()
        return loss

    def _track_epochs(self):
        self.current_iter += 1
        if self.current_iter % self.num_batches_per_epoch == 0:
            self.current_iter = 0
            self.epoch_count += 1
            self.tracking_lr.append(self.current_lr)
