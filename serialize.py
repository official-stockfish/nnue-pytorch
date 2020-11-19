import argparse
import halfkp
import math
import model as M
import numpy
import nnue_bin_dataset
import struct
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

def coalesce_weights(weights):
  # incoming weights are in [256][INPUTS]
  print('factorized shape:', weights.shape)
  k_base = 41024
  p_base = k_base + 64
  result = []
  # Goal here is to add together all the weights that would be active for the
  # given piece and king position in the halfkp inputs.
  for i in range(k_base):
    k_idx = i // halfkp.NUM_PLANES
    p_idx = i % halfkp.NUM_PLANES
    w = weights.narrow(1, i, 1).clone()
    # TODO - divide by 20 to approximate # of pieces on the board, but this is
    # a huge hack.  Issue is there is only one king position set in the factored
    # positions, but we add it's weights to the # of pieces on the board.  This
    # vastly overweights the king value.
    w = w + weights.narrow(1, k_base + k_idx, 1) / 20
    if p_idx > 0:
      w = w + weights.narrow(1, p_base + p_idx - 1, 1)
    result.append(w)
  return torch.cat(result, dim=1)

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model):
    self.buf = bytearray()

    self.write_header()
    self.int32(0x5d69d7b8) # Feature transformer hash
    self.write_feature_transformer(model.input)
    self.int32(0x63337156) # FC layers hash
    self.write_fc_layer(model.l1)
    self.write_fc_layer(model.l2)
    self.write_fc_layer(model.output, is_output=True)

  def write_header(self):
    self.int32(0x7AF32F16) # version
    self.int32(0x3e5aa6ee) # halfkp network hash
    description = b"Features=HalfKP(Friend)[41024->256x2],"
    description += b"Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
    description += b"(ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))"
    self.int32(len(description)) # Network definition
    self.buf.extend(description)

  def write_feature_transformer(self, layer):
    # int16 bias = round(x * 127)
    # int16 weight = round(x * 127)
    bias = layer.bias.data
    bias = bias.mul(127).round().to(torch.int16)
    print('ft bias:', numpy.histogram(bias.numpy()))
    self.buf.extend(bias.flatten().numpy().tobytes())

    weight = layer.weight.data
    weight = coalesce_weights(weight)
    weight = weight.mul(127).round().to(torch.int16)
    print('ft weight:', numpy.histogram(weight.numpy()))
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    self.buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
    kMaxWeight = 127.0 / kWeightScale # roughly 2.0

    # int32 bias = round(x * kBiasScale)
    # int8 weight = round(x * kWeightScale)
    bias = layer.bias.data
    bias = bias.mul(kBiasScale).round().to(torch.int32)
    print('fc bias:', numpy.histogram(bias.numpy()))
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
    print('fc weight:', numpy.histogram(weight.numpy()))
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int32(self, v):
    self.buf.extend(struct.pack("<i", v))

class NNUEReader():
  def __init__(self, f):
    self.f = f
    self.model = M.NNUE()

    self.read_header()
    self.read_int32(0x5d69d7b8) # Feature transformer hash
    self.read_feature_transformer(self.model.input)
    self.read_int32(0x63337156) # FC layers hash
    self.read_fc_layer(self.model.l1)
    self.read_fc_layer(self.model.l2)
    self.read_fc_layer(self.model.output, is_output=True)

  def read_header(self):
    self.read_int32(0x7AF32F16) # version
    self.read_int32(0x3e5aa6ee) # halfkp network hash
    desc_len = self.read_int32() # Network definition
    description = self.f.read(desc_len)

  def tensor(self, dtype, shape):
    d = numpy.fromfile(self.f, dtype, math.prod(shape))
    d = torch.from_numpy(d.astype(numpy.float32))
    d = d.reshape(shape)
    return d

  def read_feature_transformer(self, layer):
    layer.bias.data = self.tensor(numpy.int16, layer.bias.shape).divide(127.0)
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    weights = self.tensor(numpy.int16, layer.weight.shape[::-1])
    layer.weight.data = weights.divide(127.0).transpose(0, 1)

  def read_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape).divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, layer.weight.shape).divide(kWeightScale)

  def read_int32(self, expected=None):
    v = struct.unpack("<i", self.f.read(4))[0]
    if expected is not None and v != expected:
      raise Exception("Expected: %x, got %x" % (expected, v))
    return v

def test(model):
  import nnue_dataset
  dataset = 'd8_100000.bin'
  stream_cpp = nnue_dataset.SparseBatchDataset(halfkp.NAME, dataset, 1)
  stream_cpp_iter = iter(stream_cpp)
  tensors_cpp  = next(stream_cpp_iter)[:4]
  print('cpp:', tensors_cpp[3])
  print(model(*tensors_cpp))

  stream_py = nnue_bin_dataset.NNUEBinData(dataset)
  stream_py_iter = iter(stream_py)
  tensors_py = next(stream_py_iter)
  print('python:', torch.nonzero(tensors_py[3]).squeeze())
  tensors_py = [v.reshape((1,-1)) for v in tensors_py[:4]]

  weights = coalesce_weights(model.input.weight.data)
  model.input.weight = torch.nn.Parameter(weights)
  print(model(*tensors_py))

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
  parser.add_argument("target", help="Target file (can be .pt or .nnue)")
  args = parser.parse_args()

  print('Converting %s to %s' % (args.source, args.target))

  if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
    if not args.target.endswith(".nnue"):
      raise Exception("Target file must end with .nnue")
    if args.source.endswith(".pt"):
      nnue = torch.load(args.source)
    else:
      nnue = M.NNUE.load_from_checkpoint(args.source)
    nnue.eval()
    #test(nnue)
    writer = NNUEWriter(nnue)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  elif args.source.endswith(".nnue"):
    if not args.target.endswith(".pt"):
      raise Exception("Target file must end with .pt")
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f)
    torch.save(reader.model, args.target)
  else:
    raise Exception('Invalid filetypes: ' + str(args))

if __name__ == '__main__':
  main()
