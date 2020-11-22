import chess
import halfkp
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

class FeatureTransformer(nn.Module):
  def __init__(self, feature_set):
    super(FeatureTransformer, self).__init__()
    self.num_inputs = feature_set.INPUTS

    self.input = nn.Linear(self.num_inputs, L1)

    # Zero out the weights/biases for the factorized features
    # Weights stored as [256][41024]
    weights = self.input.weight.narrow(1, 0, feature_set.INPUTS - feature_set.FACTOR_INPUTS)
    weights = torch.cat((weights, torch.zeros(L1, feature_set.FACTOR_INPUTS)), dim=1)
    self.input.weight = nn.Parameter(weights)

    self.relu = nn.Hardtanh(min_val=0.0, max_val=1.0)

  def forward(self, x):
    us, them, w_in, b_in = x
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    return self.relu(l0_)

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set=halfkp, lambda_=1.0, devices=['cpu']):
    super(NNUE, self).__init__()
    num_inputs = feature_set.INPUTS

    self.devices = devices
    self.main_device = devices[0]
    self.lambda_ = lambda_
    self.feature_set = feature_set

    self.models = [nn.Sequential(
      FeatureTransformer(feature_set),
      nn.Linear(2 * L1, L2),
      nn.Hardtanh(min_val=0.0, max_val=1.0),
      nn.Linear(L2, L3),
      nn.Hardtanh(min_val=0.0, max_val=1.0),
      nn.Linear(L3, 1)).to(device) for device in self.devices]

    self.optimizer = ranger.Ranger(self.models[0].parameters())

  def forward(self, us, them, w_in, b_in):
    return self.models[0]((us, them, w_in, b_in))

  def backward(self, loss, optimizer, optimizer_idx):
    # We override backward to do nothing because we want to do
    # backward in step_end. We can't do it here because
    # PL doesn't provide a way to pass multiple losses and stuff.
    pass

  def step_(self, batches, batch_idx):
    uss, thems, whites, blacks, outcomes, scores = batches
    if len(uss) != len(self.devices):
      raise Exception('Number of batches doesn\'t match the number of devices')

    with torch.no_grad():
      for model in self.models[1:]:
        for param, main_param in zip(model.parameters(), self.models[0].parameters()):
          param.copy_(main_param, non_blocking=True)

    qs = [m((u, t, w, b)) for m, u, t, w, b in zip(self.models, uss, thems, whites, blacks)]
    ts = outcomes
    # Divide score by 600.0 to match the expected NNUE scaling factor
    ps = [(score / 600.0).sigmoid() for score in scores]

    return {
      'losses' : [self.loss_fn(q=q, t=t, p=p) / len(self.devices) for q,t,p in zip(qs, ts, ps)]
      }

  def step_end_(self, batch_parts, loss_type, do_optimization=False):
    losses = batch_parts['losses']

    if do_optimization:
      for m in self.models:
          for param in m.parameters():
              param.grad = None

      for loss in losses:
        loss.backward()

      grads_by_model = [[param.grad.to(device=self.main_device, non_blocking=True) for param in m.parameters()] for m in self.models[1:]]

      for grads in grads_by_model:
          for main_param, grad in zip(self.models[0].parameters(), grads):
              main_param.grad += grad

      self.optimizer.step()

    loss = sum([loss.to(device=self.main_device, non_blocking=True) for loss in losses])
    self.log(loss_type, loss)
    return loss

  def loss_fn(self, *, q, p, t):
    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    result  = self.lambda_ * teacher_loss    + (1.0 - self.lambda_) * outcome_loss
    entropy = self.lambda_ * teacher_entropy + (1.0 - self.lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    return loss

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx)

  def training_step_end(self, batch_parts):
    return self.step_end_(batch_parts, 'train_loss', do_optimization=True)

  def validation_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx)

  def validation_step_end(self, batch_parts):
    return self.step_end_(batch_parts, 'val_loss')

  def test_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx)

  def test_step_end(self, batch_parts):
    return self.step_end_(batch_parts, 'test_loss')

  def configure_optimizers(self):
    return None