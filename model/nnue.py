import torch
from torch import Tensor, nn
from torchmetrics import MeanMetric, MetricCollection

from .config import NNUELightningConfig
from .lambda_utils import LambdaController
from .model import NNUEModel


def _get_parameters(layers: list[nn.Module], get_biases: bool = False):
    return [
        p
        for layer in layers
        for name, p in layer.named_parameters()
        if ("bias" in name) == get_biases and p.requires_grad
    ]


def remap_tablebase_score(
    score: Tensor,
    base: float,
    scale: float,
    decay: float
) -> Tensor:
    mate_score = 32000
    max_mate_ply = 245
    tb_mate_threshold = mate_score - max_mate_ply

    abs_score = score.abs()
    is_mate = abs_score >= tb_mate_threshold

    plies = (mate_score - abs_score).clamp(min=0)
    remapped_abs = base + (decay ** plies) * scale
    remapped_score = torch.where(score < 0, -remapped_abs, remapped_abs)

    return torch.where(is_mate, remapped_score, score)


def calculate_sf_loss(scorenet, score, outcome, loss_params, actual_lambda):
    score = remap_tablebase_score(
        score,
        base=loss_params.tb_remap_base,
        scale=loss_params.tb_remap_scale,
        decay=loss_params.tb_remap_decay
    )

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


class NNUE(nn.Module):

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

        self.max_steps = 0
        if max_epoch is not None and num_batches_per_epoch is not None:
            self.max_steps = max_epoch * num_batches_per_epoch
        self.step_counter = 0

        # lazy init so `resume_from_model` with config changes works correctly
        self.optimizer_wrapper = None

        # Initialize the lambda controller
        self.lambda_scheduler = LambdaController()

        self.loss_metrics = MetricCollection(
            {
                "train_loss_epoch": MeanMetric(),
                "val_loss_epoch": MeanMetric(),
                "test_loss_epoch": MeanMetric(),
            }
        )

    # --- setup optimizers and training hooks ---
    def configure_optimizers(self):
        optimizer_config = self.config.optimizer_config
        self.optimizer_wrapper = optimizer_config.get_optimizer_wrapper()

        LRs = [optimizer_config.lr] * 10

        ft_wd = optimizer_config.ft_weight_decay
        dense_wd = optimizer_config.dense_weight_decay
        factorized_wd = optimizer_config.factorized_weight_decay

        train_params = [
            # Feature Transformer
            {
                "params": _get_parameters([self.model.input], get_biases=False),
                "lr": LRs[0],
                "weight_decay": ft_wd,
            },
            {
                "params": _get_parameters([self.model.input], get_biases=True),
                "lr": LRs[1],
                "weight_decay": 0.0,
            },
            # Dense Layer Stacks
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.weight],
                "lr": LRs[2],
                "weight_decay": factorized_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.bias],
                "lr": LRs[3],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.weight],
                "lr": LRs[4],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.bias],
                "lr": LRs[5],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.weight],
                "lr": LRs[6],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.bias],
                "lr": LRs[7],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.output.linear.weight],
                "lr": LRs[8],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.output.linear.bias],
                "lr": LRs[9],
                "weight_decay": 0.0,
            },
        ]

        return self.optimizer_wrapper.configure_optimizers(train_params)

    # --- train / eval switch ---
    def train(self, mode: bool = True):
        retval = super().train(mode)

        if self.optimizer_wrapper is not None:
            if mode:
                self.optimizer_wrapper.switch_to_train(True)
            else:
                self.optimizer_wrapper.switch_to_eval()

        return retval

    def eval(self):
        return self.train(False)

    @staticmethod
    def load_from_checkpoint(path, config, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        nnue = NNUE(config=config)
        nnue.load_state_dict(checkpoint["state_dict"])
        return nnue

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # --- hooks ---
    def on_train_epoch_start(self):
        self.optimizer_wrapper.on_train_epoch_start(self)
        if self.max_epoch is not None and self.num_batches_per_epoch is not None:
            self.max_steps = self.max_epoch * self.num_batches_per_epoch

    def on_train_epoch_end(self):
        self.optimizer_wrapper.on_train_epoch_end(self)
        self.loss_metrics["train_loss_epoch"].reset()

    def on_validation_epoch_start(self):
        self.optimizer_wrapper.on_validation_epoch_start(self)

    def on_validation_epoch_end(self):
        self.loss_metrics["val_loss_epoch"].reset()

    def on_test_epoch_start(self):
        self.optimizer_wrapper.on_test_epoch_start(self)

    def on_test_epoch_end(self):
        self.loss_metrics["test_loss_epoch"].reset()

    def on_save_checkpoint(self, checkpoint):
        self.optimizer_wrapper.on_save_checkpoint(self, checkpoint)
        self.lambda_scheduler.on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint, resuming: bool):
        self.lambda_scheduler.on_load_checkpoint(self, checkpoint, resuming=resuming)

    def on_train_batch_start(self, batch, batch_idx):
        self.optimizer_wrapper.on_train_batch_start(self, batch, batch_idx)

    # --- checkpoint state helpers ---
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)

        # torch.compile() adds _orig_mod. prefixes to parameter/buffer keys.
        # Strip them so checkpoints remain loadable by an uncompiled NNUE.
        if any("_orig_mod." in k for k in state):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

        # Lambda scheduler state (jitter_buffer is persistent=False)
        state["lambda_scheduler_state_dict"] = self.lambda_scheduler.state_dict()
        state["jitter_buffer_value"] = self.lambda_scheduler.jitter_buffer

        # Optimizer state
        if self.optimizer_wrapper is not None and getattr(self.optimizer_wrapper, "optimizer", None) is not None:
            state["optimizer_state_dict"] = self.optimizer_wrapper.optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = dict(state_dict)
        lambda_scheduler_state = state_dict.pop("lambda_scheduler_state_dict", None)
        jitter_buffer_value = state_dict.pop("jitter_buffer_value", None)
        optimizer_state = state_dict.pop("optimizer_state_dict", None)

        # If this NNUE (or a submodule) was torch.compile'd, the expected keys
        # contain _orig_mod. prefixes that plain checkpoint keys don't have.
        # Build a mapping from the plain key to the actual key and remap.
        expected = super().state_dict()
        if any("_orig_mod." in k for k in expected):
            remap = {k.replace("_orig_mod.", ""): k for k in expected}
            state_dict = {remap.get(k, k): v for k, v in state_dict.items()}

        super().load_state_dict(state_dict, *args, **kwargs)

        if lambda_scheduler_state is not None:
            self.lambda_scheduler.load_state_dict(lambda_scheduler_state)

        if jitter_buffer_value is not None:
            self.lambda_scheduler.jitter_buffer.copy_(
                jitter_buffer_value.to(
                    device=self.lambda_scheduler.jitter_buffer.device,
                    dtype=self.lambda_scheduler.jitter_buffer.dtype,
                )
            )

        if (
            optimizer_state is not None
            and self.optimizer_wrapper is not None
            and getattr(self.optimizer_wrapper, "optimizer", None) is not None
        ):
            self.optimizer_wrapper.optimizer.load_state_dict(optimizer_state)

    # --- Training step implementation ---

    def train_step(self, batch, current_epoch, global_step):
        _ = current_epoch
        loss = self.compute_loss(batch, global_step)
        self.loss_metrics["train_loss_epoch"].update(loss)
        return {"loss": loss, "train_loss": loss.detach()}

    @torch.no_grad()
    def val_step(self, batch, current_epoch, global_step):
        _ = current_epoch
        loss = self.compute_loss(batch, global_step)
        self.loss_metrics["val_loss_epoch"].update(loss)
        return {"val_loss": loss}

    @torch.no_grad()
    def test_step(self, batch, current_epoch, global_step):
        _ = current_epoch
        loss = self.compute_loss(batch, global_step)
        self.loss_metrics["test_loss_epoch"].update(loss)
        return {"test_loss": loss}

    def compute_loss(self, batch: tuple[Tensor, ...], current_step: int):
        (
            us,
            them,
            white_indices,
            black_indices,
            _outcome,
            _score,
            piece_count,
        ) = batch
        scorenet = self.model(
            us,
            them,
            white_indices,
            black_indices,
            piece_count,
            self.config.use_fake_act_quantization,
            self.config.use_fake_weight_quantization,
        )
        return self.compute_loss_with_scorenet(scorenet, batch, current_step)

    def compute_loss_with_scorenet(
        self, scorenet: Tensor, batch: tuple[Tensor, ...], current_step: int
    ):
        (
            _us,
            _them,
            _white_indices,
            _black_indices,
            outcome,
            score,
            _piece_count,
        ) = batch

        scorenet = scorenet * self.model.quantization.nnue2score

        actual_lambda = self.lambda_scheduler(
            loss_params=self.config.loss_params,
            current_step=current_step,
            max_steps=self.max_steps,
            is_training=self.training,
            scorenet=scorenet,
        )

        sf_loss = calculate_sf_loss(
            scorenet, score, outcome, self.config.loss_params, actual_lambda
        )

        return sf_loss
