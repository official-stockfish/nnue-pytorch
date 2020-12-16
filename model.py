import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

POLICY_L1 = 256
POLICY_L2 = 256
POLICY = 64 * 64 # one-hot -> from x to (ignores underpromotions)

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, lambda_=1.0):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)

    self.policy_l1 = nn.Linear(2 * L1, POLICY_L1)
    self.policy_l2 = nn.Linear(POLICY_L1, POLICY_L2)
    self.policy = nn.Linear(POLICY_L2, POLICY)

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
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_clamp = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_clamp), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    # Policy head
    p_l1 = F.relu(self.policy_l1(F.relu(l0_)))
    p_l2 = F.relu(self.policy_l2(p_l1))
    policy_ = self.policy(p_l2)

    return x, policy_

  def step_(self, batch, batch_idx, loss_type):
    us, them, white, black, outcome, score, move = batch

    q, p = self(us, them, white, black)
    # Scale score by 600.0 to match the expected NNUE scaling factor
    value_loss = F.mse_loss(q, score / 600)
    # Scale policy loss down by 5 so mse loss has a bit more weight (policy loss ~3)
    policy_loss = F.cross_entropy(p, move.long()) / 50
    self.log(loss_type + '_value_loss', value_loss)
    self.log(loss_type + '_policy_loss', policy_loss)
    return value_loss + policy_loss

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test')

  def configure_optimizers(self):
    optimizer = ranger.Ranger(self.parameters())
    return optimizer
