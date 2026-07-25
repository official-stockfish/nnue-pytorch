import os
import sys
import pytest
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import TerminateOnNaN

class DummyDataset(Dataset):
    def __len__(self):
        return 10
    def __getitem__(self, idx):
        return torch.randn(1), torch.randn(1)

class DummyModel(L.LightningModule):
    def __init__(self, produce_nan_at_step=3, validation_nan=False):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.produce_nan_at_step = produce_nan_at_step
        self.validation_nan = validation_nan

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.layer(x).sum()
        if self.global_step == self.produce_nan_at_step and not self.validation_nan:
            loss = loss * float('nan')
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.layer(x).sum()
        if self.validation_nan:
            loss = loss * float('nan')
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

@pytest.fixture
def dummy_dataloader():
    dataset = DummyDataset()
    return DataLoader(dataset, batch_size=1)

def test_nan_in_training(dummy_dataloader):
    model = DummyModel(produce_nan_at_step=3, validation_nan=False)
    
    callback = TerminateOnNaN()
    trainer = L.Trainer(
        max_epochs=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    
    trainer.fit(model, dummy_dataloader)
    assert model.global_step < 10, "Should have stopped early due to train loss NaN"
    assert trainer.should_stop is True
    assert callback.nan_detected is True

def test_nan_in_validation(dummy_dataloader):
    model = DummyModel(produce_nan_at_step=999, validation_nan=True)
    
    callback = TerminateOnNaN()
    trainer = L.Trainer(
        max_epochs=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    
    trainer.fit(model, dummy_dataloader, val_dataloaders=dummy_dataloader)
    assert trainer.should_stop is True
    assert callback.nan_detected is True
