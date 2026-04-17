import lightning as L
import torch
from torch import Tensor, nn
from torchmetrics import MeanMetric, MetricCollection

from .config import NNUELightningConfig
from .model import NNUEModel


def _get_parameters(layers: list[nn.Module], get_biases: bool = False):
    return [
        p
        for layer in layers
        for name, p in layer.named_parameters()
        if ("bias" in name) == get_biases and p.requires_grad
    ]


def sf_loss(scorenet, score, outcome, loss_params, actual_lambda):
    # convert the network and search scores to an estimate match result
    # based on the win_rate_model, with scalings and offsets optimized
    q = (scorenet - loss_params.in_offset) / loss_params.in_scaling
    qm = (-scorenet - loss_params.in_offset) / loss_params.in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

    s = (score - loss_params.out_offset) / loss_params.out_scaling
    sm = (-score - loss_params.out_offset) / loss_params.out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

    # blend that eval based score with the actual game outcome
    t = outcome

    pt = pf * actual_lambda + t * (1.0 - actual_lambda)

    # use a MSE-like loss function
    loss = torch.pow(torch.abs(pt - qf), loss_params.pow_exp)
    if loss_params.qp_asymmetry != 0.0:
        loss = loss * ((qf > pt) * loss_params.qp_asymmetry + 1)

    weights = 1 + (2.0**loss_params.w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), loss_params.w2)
    loss = (loss * weights).sum() / weights.sum()

    return loss

class MoeLoss():
    def __init__(self, logits_probe):
        self.captured_routing_data = {}
        logits_probe.register_forward_hook(self._capture_routing_hook)

    def _capture_routing_hook(self, module, input, output):
        self.captured_routing_data["routing_data"] = output

    def get_captured_data(self):
        return self.captured_routing_data.pop("routing_data")

    def moe_load_balancing_loss(self, logits, hard_weights) -> torch.Tensor:
        """
        Standard MoE auxiliary loss (Switch Transformer / Shazeer et al.)
        logits: (B, num_buckets) - The raw outputs of the router before softmax.
        hard_weights: (B, num_buckets) - The one-hot discrete routing decisions from Gumbel-Softmax.
        """
        num_buckets = logits.size(-1)

        # p_i: Mean routing probability per bucket
        probs = torch.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)

        # f_i: Fraction of batch actually routed to each bucket (detached to stop gradients)
        # hard_weights is already one-hot, so taking the mean gives the fraction per bucket
        mean_fractions = hard_weights.float().mean(dim=0).detach()

        # Scaled dot product
        moe_loss = num_buckets * torch.sum(mean_fractions * mean_probs)

        return moe_loss - 1.0  # Subtract 1 to make the loss zero when perfectly balanced (mean_fractions == mean_probs)

    @torch.no_grad()
    @torch.compiler.disable
    def get_moe_ratio(self, hard_weights: torch.Tensor) -> float:
        """
        Calculates the ratio between the most and least utilized experts.
        hard_weights: (B, num_experts) one-hot tensor
        """
        # Sum across the batch to get total tokens per expert
        expert_counts = hard_weights.float().sum(dim=0)

        max_usage = torch.max(expert_counts)
        min_usage = torch.min(expert_counts)
        min_usage = torch.clamp(min_usage, min=1e-6)

        return max_usage / min_usage


class NNUE(L.LightningModule):
    """
    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores
    """

    def __init__(
        self,
        config: NNUELightningConfig,
        max_epoch=None,
        num_batches_per_epoch=None,
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
    ):
        super().__init__()

        self.model: NNUEModel = NNUEModel(
            config.features,
            config.model_config,
            num_psqt_buckets,
            num_ls_buckets,
        )
        self.config = config
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.param_index = param_index

        # lazy init so `resume_from_model` with config changes works correctly
        self.optimizer_wrapper = None
        self.moe_loss = None

        self.loss_metrics = MetricCollection ({
            "train_loss_epoch": MeanMetric(),
            "val_loss_epoch": MeanMetric(),
            "test_loss_epoch": MeanMetric(),
            "train_moe_loss_epoch": MeanMetric(),
            "val_moe_loss_epoch": MeanMetric(),
            "test_moe_loss_epoch": MeanMetric(),
            "train_moe_ratio_epoch": MeanMetric(),
            "val_moe_ratio_epoch": MeanMetric(),
            "test_moe_ratio_epoch": MeanMetric(),
        })

    # --- setup optimizers and training hooks ---
    def configure_optimizers(self):
        if self.max_epoch is None:
            print("[NNUE] Required parameter for training not set: max_epoch")

        optimizer_config = self.config.optimizer_config
        self.optimizer_wrapper = optimizer_config.get_optimizer_wrapper(
            self.max_epoch, self.num_batches_per_epoch
        )

        LR = optimizer_config.lr
        ft_wd = optimizer_config.ft_weight_decay
        dense_wd = optimizer_config.dense_weight_decay

        train_params = [
            # Feature Transformer
            {
                "params": _get_parameters([self.model.input], get_biases=False),
                "lr": LR,
                "weight_decay": ft_wd,
            },
            {
                "params": _get_parameters([self.model.input], get_biases=True),
                "lr": LR,
                "weight_decay": 0.0,
            },
            # Dense Layer Stacks
            {
                "params": [self.model.layer_stacks.l1.linear.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.bias],
                "lr": LR,
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.bias],
                "lr": LR,
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.output.linear.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.output.linear.bias],
                "lr": LR,
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.router.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.router.bias, self.model.router_ls],
                "lr": LR,
                "weight_decay": 0.0,
            },
        ]

        return self.optimizer_wrapper.configure_optimizers(train_params)

    def setup(self, stage: str) -> None:
        self.moe_loss = MoeLoss(self.model.logits_probe)

    def on_train_epoch_start(self):
        self.optimizer_wrapper.on_train_epoch_start(self)

        for metric_key in self.loss_metrics:
            self.loss_metrics[metric_key].reset()

    def on_train_epoch_end(self):
        self.optimizer_wrapper.on_train_epoch_end(self)
        self._log_epoch_end("train")

    def on_validation_epoch_start(self):
        self.optimizer_wrapper.on_validation_epoch_start(self)

    def on_validation_epoch_end(self):
        self._log_epoch_end("val")

    def on_test_epoch_start(self):
        self.optimizer_wrapper.on_test_epoch_start(self)

    def on_test_epoch_end(self):
        self._log_epoch_end("test")

    def on_save_checkpoint(self, checkpoint):
        self.optimizer_wrapper.on_save_checkpoint(self, checkpoint)

    def on_train_batch_start(self, batch, batch_idx):
        self.optimizer_wrapper.on_train_batch_start(self, batch, batch_idx)

    def _log_epoch_end(self, loss_type):
        metrics_to_log = [
            f"{loss_type}_loss_epoch",
            f"{loss_type}_moe_loss_epoch",
            f"{loss_type}_moe_ratio_epoch",
        ]

        for metric_key in metrics_to_log:
            if metric_key not in self.loss_metrics:
                print(f"[NNUE] Warning: Metric {metric_key} not found in loss_metrics.")
                continue

            self.log(
                metric_key,
                self.loss_metrics[metric_key],
                prog_bar=False,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
            )


    # --- Training step implementation ---

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test")

    def step_(self, batch: tuple[Tensor, ...], batch_idx, loss_type):
        _ = batch_idx  # unused, but required by pytorch-lightning

        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        ) = batch

        scorenet = (
            self.model(
                us,
                them,
                white_indices,
                white_values,
                black_indices,
                black_values,
                psqt_indices,
                layer_stack_indices,
            )
            * self.model.quantization.nnue2score
        )

        loss_params = self.config.loss_params
        actual_lambda = loss_params.start_lambda + (loss_params.end_lambda - loss_params.start_lambda) * (
            self.current_epoch / self.max_epoch
        )

        fit_loss = sf_loss(scorenet, score, outcome, loss_params, actual_lambda)

        if self.moe_loss is not None:
            logits, hard_weights = self.moe_loss.get_captured_data()
            moe_loss = self.moe_loss.moe_load_balancing_loss(logits, hard_weights)
            moe_ratio = self.moe_loss.get_moe_ratio(hard_weights)
            # logging unweighted moe_loss
            self.loss_metrics[f"{loss_type}_moe_loss_epoch"].update(moe_loss.detach())
            self.loss_metrics[f"{loss_type}_moe_ratio_epoch"].update(moe_ratio.detach())
            moe_loss = loss_params.moe_loss_weight * moe_loss

        else:
            moe_loss = 0.0

        loss = fit_loss + moe_loss

        self.loss_metrics[f"{loss_type}_loss_epoch"].update(fit_loss.detach())
        self.log(
            f"{loss_type}_loss",
            fit_loss,
            prog_bar=False,
            sync_dist=False,
            on_epoch=False,
            on_step=True,
        )
        return loss
