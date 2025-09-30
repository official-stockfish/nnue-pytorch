import lightning as L
import ranger21
import torch
from torch import Tensor, nn

from .config import LossParams, ModelConfig
from .features import FeatureSet
from .model import NNUEModel
from .quantize import QuantizationConfig


def _get_parameters(layers: list[nn.Module]):
    return [p for layer in layers for p in layer.parameters()]


class NNUE(L.LightningModule):
    """
    feature_set - an instance of FeatureSet defining the input features

    lambda_ = 0.0 - purely based on game results
    0.0 < lambda_ < 1.0 - interpolated score and result
    lambda_ = 1.0 - purely based on search scores

    gamma - the multiplicative factor applied to the learning rate after each epoch

    lr - the initial learning rate
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        config: ModelConfig,
        quantize_config: QuantizationConfig,
        max_epoch=800,
        num_batches_per_epoch=int(100_000_000 / 16384),
        gamma=0.992,
        lr=8.75e-4,
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
        loss_params=LossParams(),
    ):
        super().__init__()
        self.model: NNUEModel = NNUEModel(
            feature_set, config, quantize_config, num_psqt_buckets, num_ls_buckets
        )
        self.loss_params = loss_params
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.gamma = gamma
        self.lr = lr
        self.param_index = param_index

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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
        loss = loss.mean()

        self.log(loss_type, loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

    def configure_optimizers(self):
        LR = self.lr
        train_params = [
            {"params": _get_parameters([self.model.input]), "lr": LR, "gc_dim": 0},
            {"params": [self.model.layer_stacks.l1.factorized_linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l1.factorized_linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.l1.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l1.linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.l2.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.l2.linear.bias], "lr": LR},
            {"params": [self.model.layer_stacks.output.linear.weight], "lr": LR},
            {"params": [self.model.layer_stacks.output.linear.bias], "lr": LR},
        ]

        optimizer = ranger21.Ranger21(
            train_params,
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1.0e-7,
            using_gc=False,
            using_normgc=False,
            weight_decay=0.0,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_epochs=self.max_epoch,
            warmdown_active=False,
            use_warmup=False,
            use_adaptive_gradient_clipping=False,
            softplus=False,
            pnm_momentum_factor=0.0,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )

        return [optimizer], [scheduler]
