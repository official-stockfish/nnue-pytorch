import chess
import halfkp
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

L1 = 256

def cp_conversion(x, alpha=0.0016):
  return (x * alpha).sigmoid()

class NNUE(pl.LightningModule):
  def __init__(self):
    super(NNUE, self).__init__()
    self.w_input = nn.Linear(halfkp.INPUTS, L1)
    self.b_input = nn.Linear(halfkp.INPUTS, L1)
    self.l1 = nn.Linear(2*L1, 32)
    self.l2 = nn.Linear(32, 32)
    self.output = nn.Linear(32, 1)

  def forward(self, x):
    turn, w_in, b_in = x
    w = self.w_input(w_in)
    b = self.b_input(b_in)
    l0_ = F.relu(turn * torch.cat([w, b], dim=1) + (1.0 - turn) * torch.cat([b, w], dim=1))
    l1_ = F.relu(self.l1(l0_))
    l2_ = F.relu(self.l2(l1_))
    return self.output(l2_)

  def training_step(self, batch, batch_idx):
    turn, white, black, outcome, score = batch
    output = self((turn, white, black))
    loss = F.mse_loss(output, cp_conversion(score))
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    turn, white, black, outcome, score = batch
    output = self((turn, white, black))
    loss = F.mse_loss(output, cp_conversion(score))
    self.log('val_loss', loss)

  def configure_optimizers(self):
    optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0)
    return optimizer
