from collections import namedtuple
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

QTensor = namedtuple('QTensor', ['tensor', 'scale'])
def quantize_tensor(x, dtype=torch.int8, scale=127.0, min_max=None):
  if min_max is not None:
    q_x = x.clamp(min_max[0], min_max[1])
  else:
    q_x = x
  q_x = (q_x * scale).round().to(dtype)
  return QTensor(tensor=q_x, scale=scale)

def dequantize_tensor(q_x):
  return q_x.tensor.float() / q_x.scale

class FakeQuantOp(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, dtype=torch.int8, scale=127.0, min_max=None):
    x = quantize_tensor(x, dtype=dtype, scale=scale, min_max=min_max)
    x = dequantize_tensor(x)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    # straight through estimator
    return grad_output, None, None, None

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores
  """
  def __init__(self, feature_set, lambda_=1.0):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_

    self._zero_virtual_feature_weights()

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    for a, b in self.feature_set.get_virtual_feature_ranges():
      weights[a:b, :] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in):
    return self.quant_forward(us, them, w_in, b_in)

    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def quant_fc(self, layer, x, is_output=False):
    l_w = layer.weight.data
    l_b = layer.bias.data
    if not is_output:
      bias_scale = 8128
    else:
      bias_scale = 9600
    layer.weight.data = FakeQuantOp.apply(l_w, torch.int8, bias_scale / 127.0, (-127.0 / 64.0, 127.0 / 64.0))
    layer.bias.data = FakeQuantOp.apply(l_b, torch.int32, bias_scale)
    result = layer(x)
    layer.weight.data = l_w
    layer.bias.data = l_b
    return result

  def quant_forward(self, us, them, w_in, b_in):
    input_w = self.input.weight.data
    input_b = self.input.bias.data
    self.input.weight.data = FakeQuantOp.apply(input_w, torch.int16, 127.0)
    self.input.bias.data = FakeQuantOp.apply(input_b, torch.int16, 127.0)
    w = self.input(w_in)
    b = self.input(b_in)
    self.input.weight.data = input_w
    self.input.bias.data = input_b

    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.quant_fc(self.l1, l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.quant_fc(self.l2, l1_), 0.0, 1.0)
    x = self.quant_fc(self.output, l2_, is_output=True)
    return x

  def step_(self, batch, batch_idx, loss_type):
    us, them, white, black, outcome, score = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    nnue2score = 600
    scaling = 361

    q = self(us, them, white, black) * nnue2score / scaling
    t = outcome
    p = (score / scaling).sigmoid()

    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    result  = self.lambda_ * teacher_loss    + (1.0 - self.lambda_) * outcome_loss
    entropy = self.lambda_ * teacher_entropy + (1.0 - self.lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    self.log(loss_type, loss)
    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def configure_optimizers(self):
    # increasing the eps leads to less saturated nets with a few dead neurons
    optimizer = ranger.Ranger(self.parameters(),betas=(.9, 0.999), eps=1.0e-7)
    # Drop learning rate after 75 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.3)
    return [optimizer], [scheduler]
