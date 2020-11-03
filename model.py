import chess
import halfkp
import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
import torch.nn.functional as F
import pytorch_lightning as pl

L1 = 256
L2 = 32
L3 = 32

def cp_conversion(x, alpha=0.0016):
  return (x * alpha).sigmoid()

class NNUE(pl.LightningModule):
  def __init__(self):
    super(NNUE, self).__init__()
    self.input = nn.Linear(halfkp.INPUTS, L1)
    self.input_act = nn.ReLU()
    self.l1 = nn.Linear(2 * L1, L2)
    self.l1_act = nn.ReLU()
    self.l2 = nn.Linear(L2, L3)
    self.l2_act = nn.ReLU()
    self.output = nn.Linear(L3, 1)
    self.quant = QuantStub()
    self.dequant = DeQuantStub()

  def forward(self, turn, w_in, b_in):
    turn = self.quant(turn)
    w_in = self.quant(w_in)
    b_in = self.quant(b_in)
    w = self.input(w_in)
    b = self.input(b_in)
    if turn.data[0] == 1:
      l0_ = torch.cat([w, b], dim=1)
    else:
      l0_ = torch.cat([b, w], dim=1)
    l0_ = self.input_act(l0_)
    l1_ = self.l1_act(self.l1(l0_))
    l2_ = self.l2_act(self.l2(l1_))
    x = self.output(l2_)
    x = self.dequant(x)
    return x

  def step_(self, batch, batch_idx, loss_type):
    turn, white, black, outcome, score = batch
    output = self(turn, white, black)
    loss = F.mse_loss(output, cp_conversion(score))
    self.log(loss_type, loss)
    return loss

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def configure_optimizers(self):
    optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0)
    return optimizer
