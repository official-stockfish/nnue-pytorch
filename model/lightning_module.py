import lightning as L
import torch
from torch import Tensor, nn

from .config import LossParams, ModelConfig
from .model import NNUEModel
from .quantize import QuantizationConfig

from .optimizers.ranger21_wrapper import Ranger21Wrapper
from .optimizers.schedulefree_wrapper import ScheduleFreeWrapper

def _get_parameters(layers: list[nn.Module], get_biases: bool = False):
    return [
        p
        for layer in layers
        for name, p in layer.named_parameters()
        if ("bias" in name) == get_biases and p.requires_grad
    ]


class NNUE(L.LightningModule):
    """
    feature_name - a string identifying the feature transformer (e.g. "HalfKAv2_hm")

    optimizer_name - a string identifying the optimizer wrapper ("schedulefree" or "ranger21")

    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores

    gamma - the multiplicative factor applied to the learning rate after each epoch

    lr - the initial learning rate
    """

    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        optimizer_name: str = "ranger21",
        max_epoch=800,
        num_batches_per_epoch=int(100_000_000 / 16384),
        gamma=0.992,
        lr=8.75e-4,
        warmup_steps=10000,
        ft_weight_decay=0.0,
        dense_weight_decay=0.0,
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
        loss_params=LossParams(),
        **kwargs,
    ):
        super().__init__()

        self.model: NNUEModel = NNUEModel(
            feature_name, config, quantize_config, num_psqt_buckets, num_ls_buckets
        )
        self.loss_params = loss_params
        self.max_epoch = max_epoch
        self.num_batches_per_epoch=num_batches_per_epoch
        self.lr = lr
        self.dense_weight_decay = dense_weight_decay
        self.ft_weight_decay = ft_weight_decay
        self.param_index = param_index

        optimizer_name = optimizer_name.lower().strip()
        if optimizer_name == "schedulefree":
            self.optimizer_wrapper = ScheduleFreeWrapper(
                lr=lr,
                warmup_steps=warmup_steps,
                **kwargs
            )
        elif optimizer_name == "ranger21":
            self.optimizer_wrapper = Ranger21Wrapper(
                max_epoch=max_epoch,
                gamma=gamma,
                num_batches_per_epoch=num_batches_per_epoch,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer_name: '{optimizer_name}'. Expected 'schedulefree' or 'ranger21'.")

        if self.dense_weight_decay > 0.0 or self.ft_weight_decay > 0.0:
            print(f"Using weight decay - ft_weight_decay: {self.ft_weight_decay}, dense_weight_decay: {self.dense_weight_decay}")

    # --- setup optimizers and training hooks ---

    def configure_optimizers(self):
        LR = self.lr

        train_params = [
            # Feature Transformer
            {"params": _get_parameters([self.model.input], get_biases=False), "lr": LR, "weight_decay": self.ft_weight_decay},
            {"params": _get_parameters([self.model.input], get_biases=True), "lr": LR, "weight_decay": 0.0},

            # Dense Layer Stacks
            {"params": [self.model.layer_stacks.l1.factorized_linear.weight], "lr": LR, "weight_decay": self.dense_weight_decay},
            {"params": [self.model.layer_stacks.l1.factorized_linear.bias], "lr": LR, "weight_decay": 0.0},
            {"params": [self.model.layer_stacks.l1.linear.weight], "lr": LR, "weight_decay": self.dense_weight_decay},
            {"params": [self.model.layer_stacks.l1.linear.bias], "lr": LR, "weight_decay": 0.0},
            {"params": [self.model.layer_stacks.l2.linear.weight], "lr": LR, "weight_decay": self.dense_weight_decay},
            {"params": [self.model.layer_stacks.l2.linear.bias], "lr": LR, "weight_decay": 0.0},
            {"params": [self.model.layer_stacks.output.linear.weight], "lr": LR, "weight_decay": self.dense_weight_decay},
            {"params": [self.model.layer_stacks.output.linear.bias], "lr": LR, "weight_decay": 0.0},
        ]

        return self.optimizer_wrapper.configure_optimizers(train_params)

    def on_train_epoch_start(self):
        self.optimizer_wrapper.on_train_epoch_start(self)

    def on_train_epoch_end(self):
        self.optimizer_wrapper.on_train_epoch_end(self)

    def on_validation_epoch_start(self):
        self.optimizer_wrapper.on_validation_epoch_start(self)

    def on_test_epoch_start(self):
        self.optimizer_wrapper.on_test_epoch_start(self)

    def on_save_checkpoint(self, checkpoint):
        self.optimizer_wrapper.on_save_checkpoint(self, checkpoint)

    def on_train_batch_start(self, batch, batch_idx):
        self.optimizer_wrapper.on_train_batch_start(self, batch, batch_idx)

    # --- Training step implementation ---

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

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
            )
            * self.model.quantization.nnue2score
        )

        p = self.loss_params
        # convert the network and search scores to an estimate match result
        # based on the win_rate_model, with scalings and offsets optimized
        q = (scorenet - p.in_offset) / p.in_scaling
        qm = (-scorenet - p.in_offset) / p.in_scaling
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

        s = (score - p.out_offset) / p.out_scaling
        sm = (-score - p.out_offset) / p.out_scaling
        pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

        # blend that eval based score with the actual game outcome
        t = outcome
        actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * (
            self.current_epoch / self.max_epoch
        )
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        # use a MSE-like loss function
        loss = torch.pow(torch.abs(pt - qf), p.pow_exp)
        if p.qp_asymmetry != 0.0:
            loss = loss * ((qf > pt) * p.qp_asymmetry + 1)

        weights = 1 + (2.0**p.w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), p.w2)
        loss = (loss * weights).sum() / weights.sum()

        self.log(loss_type, loss, prog_bar=True, sync_dist=True)

        return loss
