import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from feature_transformer import DoubleFeatureTransformerSlice

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

def coalesce_ft_weights(model, layer):
  weight = layer.weight.data
  indices = model.feature_set.get_virtual_to_real_features_gather_indices()
  weight_coalesced = weight.new_zeros((model.feature_set.num_real_features, weight.shape[1]))
  for i_real, is_virtual in enumerate(indices):
    weight_coalesced[i_real, :] = sum(weight[i_virtual, :] for i_virtual in is_virtual)
  return weight_coalesced

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, lambda_=1.0):
    super(NNUE, self).__init__()
    self.input = DoubleFeatureTransformerSlice(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_

    self._zero_virtual_feature_weights()
    self._correct_init_biases()

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
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[a:b, :] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  Pytorch initializes biases around 0, but we want them
  to be around activation range. Also the bias for the output
  layer should always be 0.
  '''
  def _correct_init_biases(self):
    input_bias = self.input.bias
    l1_bias = self.l1.bias
    l2_bias = self.l2.bias
    output_bias = self.output.bias
    with torch.no_grad():
      input_bias.add_(0.5)
      l1_bias.add_(0.5)
      l2_bias.add_(0.5)
      output_bias.fill_(0.0)
    self.input.bias = nn.Parameter(input_bias)
    self.l1.bias = nn.Parameter(l1_bias)
    self.l2.bias = nn.Parameter(l2_bias)
    self.output.bias = nn.Parameter(output_bias)

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
      padding = weights.new_zeros((new_feature_block.num_virtual_features, weights.shape[1]))
      weights = torch.cat([weights, padding], dim=0)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, white_indices, white_values, black_indices, black_values):
    w, b = self.input(white_indices, white_values, black_indices, black_values)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def step_(self, batch, batch_idx, loss_type):
    us, them, white_indices, white_values, black_indices, black_values, outcome, score = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    nnue2score = 600
    in_scaling = 410
    out_scaling = 361

    q = self(us, them, white_indices, white_values, black_indices, black_values) * nnue2score / scaling
    t = outcome
    p = (score / in_scaling).sigmoid()

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
    # Train with a lower LR on the output layer
    LR = 1e-3
    train_params = [
      {'params' : self.get_specific_layers([self.input]), 'lr' : LR, 'min_weight' : -(2**15-1)/127, 'max_weight' : (2**15-1)/127 },
      {'params' : self.get_specific_layers([self.l1, self.l2]), 'lr' : LR, 'min_weight' : -127/64, 'max_weight' : 127/64 },
      {'params' : self.get_specific_layers([self.output]), 'lr' : LR / 10, 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
    ]
    # increasing the eps leads to less saturated nets with a few dead neurons
    optimizer = ranger.Ranger(train_params, betas=(.9, 0.999), eps=1.0e-7)
    # Drop learning rate after 75 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.3)
    return [optimizer], [scheduler]

  def get_specific_layers(self, layers):
    pred = lambda x: x in layers
    return self.get_layers(pred)

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        for p in i.parameters():
          if p.requires_grad:
            yield p
