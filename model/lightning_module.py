import lightning as L
import torch
import math

from torch import Tensor, nn
from torchmetrics import MeanMetric, MetricCollection

from .config import NNUELightningConfig
from .model import NNUEModel
from .quantize import QuantizationConfig


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


def ft_act_loss(activation_data, l1_weight, l2_weight, group_size=4):
    original_shape = activation_data.shape
    if original_shape[-1] % group_size != 0:
        raise ValueError(f"Feature dimension {original_shape[-1]} not divisible by group_size {group_size}")

    grouped_data = activation_data.view(*original_shape[:-1], -1, group_size)
    group_max = torch.max(grouped_data, dim=-1).values

    loss_l1 = torch.mean(nn.functional.relu(group_max + 0.1))
    loss_l2 = torch.mean(activation_data**2)

    return (l1_weight * loss_l1) + (l2_weight * loss_l2)


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
        quantize_config=QuantizationConfig(),
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
    ):
        super().__init__()

        self.model: NNUEModel = NNUEModel(
            config.features,
            config.model_config,
            quantize_config,
            num_psqt_buckets,
            num_ls_buckets,
        )
        self.config = config
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.param_index = param_index

        # lazy init so `resume_from_model` with config changes works correctly
        self.optimizer_wrapper = None

        self.loss_metrics = MetricCollection(
            {
                "train_loss_epoch": MeanMetric(),
                "val_loss_epoch": MeanMetric(),
                "test_loss_epoch": MeanMetric(),
            }
        )

        #regularization loss
        l1_w = config.loss_params.ft_activation_l1
        l2_w = config.loss_params.ft_activation_l2
        self.use_ft_activation_loss = l1_w > 0 or l2_w > 0

        # register jitter buffer
        self.register_buffer("jitter_buffer", torch.zeros(1), persistent=False)

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
                "params": [self.model.layer_stacks.l1.factorized_linear.weight],
                "lr": LR,
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.bias],
                "lr": LR,
                "weight_decay": 0.0,
            },
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
        ]

        return self.optimizer_wrapper.configure_optimizers(train_params)

    def on_train_start(self):
        self.train()

    def on_train_epoch_start(self):
        self.optimizer_wrapper.on_train_epoch_start(self)

    def on_train_epoch_end(self):
        self.optimizer_wrapper.on_train_epoch_end(self)
        self._log_epoch_end("train_loss_epoch")

    def on_validation_epoch_start(self):
        self.optimizer_wrapper.on_validation_epoch_start(self)

    def on_validation_epoch_end(self):
        self._log_epoch_end("val_loss_epoch")

    def on_test_epoch_start(self):
        self.optimizer_wrapper.on_test_epoch_start(self)

    def on_test_epoch_end(self):
        self._log_epoch_end("test_loss_epoch")

    def on_save_checkpoint(self, checkpoint):
        self.optimizer_wrapper.on_save_checkpoint(self, checkpoint)
        checkpoint["jitter_buffer_value"] = self.jitter_buffer

    def on_load_checkpoint(self, checkpoint):
        trainer = self.__dict__.get("_trainer", None)
        is_resuming = (
            trainer is not None and
            getattr(trainer, "ckpt_path", None) is not None
        )
        if is_resuming:
            if "jitter_buffer_value" in checkpoint:
                self.jitter_buffer.copy_(checkpoint["jitter_buffer_value"])

    def on_train_batch_start(self, batch, batch_idx):
        self.optimizer_wrapper.on_train_batch_start(self, batch, batch_idx)

    def _log_epoch_end(self, loss_type):
        self.log(
            f"{loss_type}",
            self.loss_metrics[f"{loss_type}"],
            prog_bar=False,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

    # --- train / eval switch ---
    def train(self, mode: bool = True):
        retval = super().train(mode)

        if hasattr(self, '_trainer') and self._trainer and self.trainer.optimizers:
            for opt in self.trainer.optimizers:
                if mode:
                    if hasattr(opt, 'train') and callable(opt.train):
                        opt.train()
                else:
                    if hasattr(opt, 'eval') and callable(opt.eval):
                        opt.eval()

        return retval


    def eval(self):
        return self.train(False)

    # --- Training step implementation ---

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

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
                return_activations=self.use_ft_activation_loss,
            )
        )
        if self.use_ft_activation_loss:
            scorenet, ft_activation = scorenet
        scorenet = scorenet * self.model.quantization.nnue2score

        loss_params = self.config.loss_params

        actual_lambda = loss_params.start_lambda + (loss_params.end_lambda - loss_params.start_lambda) * (
            self.current_epoch / self.max_epoch
        )

        if self.training:
            # Normalizing jitter_lambda_batch so that combined with decay,
            # the effective jitter magnitude remains consistent across different decay rates.
            jitter_lambda_batch = loss_params.jitter_lambda_batch * math.sqrt(1 - loss_params.jitter_decay_lambda_batch ** 2)
            batch_jitter_delta = jitter_lambda_batch * torch.randn_like(self.jitter_buffer)
            self.jitter_buffer.mul_(loss_params.jitter_decay_lambda_batch).add_(batch_jitter_delta)
            batch_jitter = self.jitter_buffer.expand_as(scorenet)
        else:
            # During evaluating, we effectively use decay = 0.0.
            batch_jitter = loss_params.jitter_lambda_batch * torch.randn_like(scorenet)
        sample_jitter = scorenet.new_empty(scorenet.shape).normal_(0, 1) * loss_params.jitter_lambda_sample
        actual_lambda = actual_lambda + batch_jitter + sample_jitter
        actual_lambda = actual_lambda.clamp(0.0, 1.0)

        fit_loss = sf_loss(scorenet, score, outcome, loss_params, actual_lambda)
        if self.use_ft_activation_loss:
            reg_loss = ft_act_loss(ft_activation, loss_params.ft_activation_l1, loss_params.ft_activation_l2)
        else:
            reg_loss = torch.zeros_like(fit_loss)
        loss = fit_loss + reg_loss

        self.loss_metrics[f"{loss_type}_epoch"].update(fit_loss)
        self.log(
            loss_type,
            fit_loss,
            prog_bar=False,
            sync_dist=False,
            on_epoch=False,
            on_step=True,
        )
        return loss
