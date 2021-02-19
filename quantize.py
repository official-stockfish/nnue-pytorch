import features
import model
import qmodel as M
import nnue_bin_dataset
import numpy
import pytorch_lightning as pl
import struct
import torch
from metrics import compute_mse
from torch.utils.data import DataLoader
from torch import nn

def coalesce_ft_weights(model, layer):
  weight = layer.weight.data
  indices = model.feature_set.get_virtual_to_real_features_gather_indices()
  weight_coalesced = weight.new_zeros((weight.shape[0], model.feature_set.num_real_features))
  for i_real, is_virtual in enumerate(indices):
    weight_coalesced[:, i_real] = sum(weight[:, i_virtual] for i_virtual in is_virtual)
  return weight_coalesced

# hardcoded for now
VERSION = 0x7AF32F16

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model):
    self.buf = bytearray()

    fc_hash = self.fc_hash(model)
    self.write_header(model, fc_hash)
    self.uint32(model.feature_set.hash ^ (M.L1*2)) # Feature transformer hash
    self.write_feature_transformer(model)
    self.uint32(fc_hash) # FC layers hash
    self.write_fc_layer(model.l1, model.input)
    self.write_fc_layer(model.l2, model.l1)
    self.write_fc_layer(model.l3, model.l2)
    self.write_fc_layer(model.output, model.l3, is_output=True)

  @staticmethod
  def fc_hash(model):
    # InputSlice hash
    prev_hash = 0xEC42E90D
    prev_hash ^= (M.L1 * 2)

    # Fully connected layers
    layers = [model.l1, model.l2, model.l3, model.output]
    for layer in layers:
      layer_hash = 0xCC03DAE4
      layer_hash += layer.out_features
      layer_hash ^= prev_hash >> 1
      layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
#      if layer.out_features != 1:
        # Clipped ReLU hash
#        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
      prev_hash = layer_hash
    return layer_hash

  def write_header(self, model, fc_hash):
    self.uint32(VERSION) # version
    self.uint32(fc_hash ^ model.feature_set.hash ^ (M.L1*2)) # halfkp network hash
    description = b"Features=HalfKP(Friend)[41024->256x2],"
    description += b"Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
    description += b"(ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
    self.uint32(len(description)) # Network definition
    self.buf.extend(description)

  def write_quant_params(self, layer, prev):
    # layer.scale - activation scale
    # layer.zero_point - activation zero point
    # layer.weight().q_scale() - weight scale
    # layer.weight().q_zero_point() - weight zero point
    # input ones come from the activations of the previous layer.
    input_scale = 1.0 if prev is None else prev.scale
    weight_scale = layer.weight().q_scale()
    output_scale = layer.scale

    input_zero_point = 0 if prev is None else prev.zero_point
    weight_zero_point = layer.weight().q_zero_point()
    output_zero_point = layer.zero_point

    scale_float = input_scale * weight_scale / output_scale
    assert(scale_float >= 0 and scale_float <= 1.0)

    scale_bits = 31
    scale = int(scale_float * (2**scale_bits))
    #print('scale_float:%f scale:%d 1.0test:%d 1.0ftest:%f' % (scale_float, scale, (255 * scale) >> scale_bits, 255 * scale_float))

    self.int32(scale)
    self.int32(scale_bits)
    self.int32(input_zero_point)
    self.int32(weight_zero_point)
    self.int32(output_zero_point)

    # activation min is zero point.
    self.int32(output_zero_point)
    self.int32(255)

    bias_scale = input_scale * weight_scale
    return bias_scale

  def write_feature_transformer(self, model):
    layer = model.input
    bias_scale = self.write_quant_params(layer, None)

    bias = (layer.bias().data * bias_scale).round().to(torch.int32)
    self.buf.extend(bias.flatten().numpy().tobytes())

    weight = layer.weight().data.int_repr()
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    self.buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())

  def write_fc_layer(self, layer, prev, is_output=False):
    bias_scale = self.write_quant_params(layer, prev)

    bias = (layer.bias().data * bias_scale).round().to(torch.int32)
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight().data.int_repr()
    # FC inputs are padded to 32 elements for simd.
    num_input = weight.shape[1]
    if num_input % 32 != 0:
      num_input += 32 - (num_input % 32)
      new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
      new_w[:, :weight.shape[1]] = weight
      weight = new_w
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int32(self, v):
    self.buf.extend(struct.pack("<i", v))

  def uint32(self, v):
    self.buf.extend(struct.pack("<I", v))

def qmodel_from_model(baseline):
  halfkp = features.get_feature_set_from_name('HalfKP')
  qm = M.NNUE(halfkp)
  qm.eval()
  layers = ['input', 'l1', 'l2', 'l3', 'output']
  for name in layers:
    setattr(qm, name, getattr(baseline, name))
  qm.input.weight = nn.Parameter(coalesce_ft_weights(baseline, baseline.input))
  qm.input.in_features = halfkp.num_real_features
  return qm

def get_loader(feature_set_name):
  halfkp = features.get_feature_set_from_name(feature_set_name)
  train = nnue_bin_dataset.NNUEBinData('large_gensfen_multipvdiff_100_d9.bin', halfkp)
  return DataLoader(torch.utils.data.Subset(train, range(0, 1024)), batch_size=64)

def load_model(ckpt_name):
  # Hardcoded to convert from HalfKP^ right now...
  factorized = features.get_feature_set_from_name('HalfKP^')
  nnue = model.NNUE.load_from_checkpoint(ckpt_name, map_location='cpu', feature_set=factorized)
  nnue.eval()
  return nnue

def main():
  trainer = pl.Trainer(progress_bar_refresh_rate=0)
  baseline = load_model('epoch279_3layer.ckpt')
  #print('baseline:', trainer.test(baseline, get_loader('HalfKP^'), verbose=False))

  nnue = qmodel_from_model(baseline)
  loader = get_loader('HalfKP')
  #print('converted to quantized net, and fused factorizer:', trainer.test(nnue, loader, verbose=False))

  fuse_layers = [
    ['input', 'input_act'],
    ['l1', 'l1_act'],
    ['l2', 'l2_act'],
  ]
  torch.quantization.fuse_modules(nnue, fuse_layers, inplace=True)

  # fbgemm config uses per-channel quantization, which is more accurate, but will be more implementation work.
  #nnue.qconfig = torch.quantization.get_default_qconfig('fbgemm')

  # HistogramObserver is most accurate, but can only be used for static quantization.
  #   reduce_range=True means the quantization is limited to 0..127 - this is useful for SSE optimizations.
  nnue.qconfig = torch.quantization.QConfig(
      activation=torch.quantization.HistogramObserver.with_args(reduce_range=True),
      weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine))
  nnue_prep = torch.quantization.prepare(nnue)
  print('fused net:', trainer.test(nnue_prep, loader, verbose=False))

  nnue_int8 = torch.quantization.convert(nnue_prep)
  #print('quantized net:', trainer.test(nnue_int8, loader, verbose=False))

  writer = NNUEWriter(nnue_int8)
  with open('quantized.nnue', 'wb') as f:
    f.write(writer.buf)
  #torch.save(nnue_int8, 'quantized_int8.pt')

  #torch.jit.save(torch.jit.script(nnue_int8), 'quantized.pt')
  #nnueq = torch.jit.load('quantized.pt')

if __name__ == '__main__':
  main()
