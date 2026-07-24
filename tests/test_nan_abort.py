from types import SimpleNamespace

import pytest
import torch

from trainer.callbacks import TerminateOnNaN


def _make_trainer(world_size=1, rank=0, device="cpu"):
    return SimpleNamespace(
        world_size=world_size,
        rank=rank,
        device=torch.device(device),
        should_stop=False,
        callback_metrics={},
    )


def test_finite_train_loss_does_not_stop():
    trainer = _make_trainer()
    callback = TerminateOnNaN()
    callback.on_train_batch_end(trainer, outputs={"loss": torch.tensor(1.23)})
    assert not trainer.should_stop
    assert not callback.nan_detected


def test_nan_train_loss_stops_training():
    trainer = _make_trainer()
    callback = TerminateOnNaN()
    callback.on_train_batch_end(trainer, outputs={"loss": torch.tensor(float("nan"))})
    assert trainer.should_stop
    assert callback.nan_detected


def test_inf_train_loss_stops_training():
    trainer = _make_trainer()
    callback = TerminateOnNaN()
    callback.on_train_batch_end(trainer, outputs={"loss": torch.tensor(float("inf"))})
    assert trainer.should_stop
    assert callback.nan_detected


def test_nan_validation_loss_stops_training():
    trainer = _make_trainer()
    callback = TerminateOnNaN()
    trainer.callback_metrics["val_loss_epoch"] = float("nan")
    callback.on_validation_epoch_end(trainer)
    assert trainer.should_stop
    assert callback.nan_detected


def test_missing_loss_is_ignored():
    trainer = _make_trainer()
    callback = TerminateOnNaN()
    callback.on_train_batch_end(trainer, outputs={})
    assert not trainer.should_stop
    assert not callback.nan_detected


def test_scalar_loss_supported():
    trainer = _make_trainer()
    callback = TerminateOnNaN()
    callback.on_train_batch_end(trainer, outputs={"loss": float("nan")})
    assert trainer.should_stop
    assert callback.nan_detected


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed not available"
)
def test_distributed_nan_aggregates_across_ranks():
    """Sanity-check the DDP aggregation helper without launching processes."""
    # This test only exercises the local branch; real multi-rank NaN handling
    # is covered by the GPU training pipeline.
    trainer = _make_trainer(world_size=2, rank=0)
    callback = TerminateOnNaN()
    callback.on_train_batch_end(trainer, outputs={"loss": torch.tensor(float("nan"))})
    assert trainer.should_stop
    assert callback.nan_detected
