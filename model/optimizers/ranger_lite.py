# RangerLite Optimizer (2026)
# A refactored, unbloated, and bug-fixed derivative of Ranger21.
#
# Original Ranger21 implementation by @lessw2020:
# URL: https://github.com/lessw2020/Ranger21
#
# Modifications and Refactoring by @TonyCongqianWang

import torch
import math
import collections

class RangerLite(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-7,
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
            pnm_momentum=pnm_momentum
        )
        super().__init__(params, defaults)

        self.lookahead_active = lookahead_active
        self.lookahead_mergetime = lookahead_mergetime
        self.lookahead_alpha = lookahead_blending_alpha
        self.lookahead_step = 0

        self.normloss_active = normloss_active
        self.normloss_factor = normloss_factor
        self.eps = eps
        self.param_size = 0

        self.use_legacy_scoping_bug = use_legacy_scoping_bug

    def unit_norm(self, x):
        """Axis-based Euclidean norm for normloss."""
        keepdim = True
        xlen = len(x.shape)
        if xlen <= 1:
            keepdim = False
            dim = None
        elif xlen in (2, 3):
            dim = 1
        elif xlen == 4:
            dim = (1, 2, 3)
        else:
            dim = tuple([i for i in range(1, xlen)])
        return x.norm(dim=dim, keepdim=keepdim, p=2.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None and isinstance(closure, collections.abc.Callable):
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        variance_ma_sum = torch.zeros(1)
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

        variance_normalized = torch.sqrt(variance_ma_sum / self.param_size)

        # Phase 2: Apply weight decay and update weights
        for group in self.param_groups:
            decay = group["weight_decay"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            pnm_factor = group["pnm_momentum"]

            # --- LEGACY BUG BEHAVIOR ---
            # This correctly targets the last parameter of the PREVIOUS group
            # (or the absolute last parameter of the network for Group 0)
            if self.use_legacy_scoping_bug and leaked_p is not None:
                if decay:
                    leaked_p.data.mul_(1 - decay * lr / variance_normalized)
                if self.normloss_active:
                    unorm = self.unit_norm(leaked_p.data)
                    corr = 2 * self.normloss_factor * (1 - torch.div(1, unorm + self.eps))
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
                        corr = 2 * self.normloss_factor * (1 - torch.div(1, unorm + self.eps))
                        p.data.mul_(1 - lr * corr)
                # ------------------------

                grad = p.grad
                step = state["step"]

                # PNM Adam Core Setup
                grad_ma = state["grad_ma"]
                variance_ma = state["variance_ma"]

                if step % 2 == 1:
                    grad_ma, neg_grad_ma = state["grad_ma"], state["neg_grad_ma"]
                else:
                    grad_ma, neg_grad_ma = state["neg_grad_ma"], state["grad_ma"]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Despite the comment. Ranger21 doesnt actually use variance_ma_max for denominator
                denom = (variance_ma.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                # Update grad_ma
                grad_ma.mul_(beta1 ** 2).add_(grad, alpha=1 - beta1 ** 2)

                # Refactored PNM calculation
                noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
                pnm_val = (
                    grad_ma.mul(1 + pnm_factor)
                    .add(neg_grad_ma, alpha=-pnm_factor)
                    .mul(1 / noise_norm)
                )
                step_size = lr / bias_correction1
                p.addcdiv_(pnm_val, denom, value=-step_size)

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

    def swap_for_inference(self):
        """Safely loads slow weights for eval/saving. Idempotent."""
        if not self.lookahead_active:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "lookahead_params" in state:
                    # Only backup if we haven't already backed up
                    if "backup_params" not in state:
                        state["backup_params"] = torch.clone(p.data)
                    p.data.copy_(state["lookahead_params"])

    def restore_for_training(self):
        """Restores fast weights for training. Idempotent."""
        if not self.lookahead_active:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "backup_params" in state:
                    p.data.copy_(state["backup_params"])
                    del state["backup_params"]
