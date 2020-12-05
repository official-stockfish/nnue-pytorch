import argparse
import halfkp
import math
import model as M
import numpy
import nnue_bin_dataset
import nnue_dataset
import struct
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import reduce
import operator

def ascii_hist(name, x, bins=6):
  N,X = numpy.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

class NNUEWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model):
    self.buf = bytearray()

    self.write_header()
    self.int32(0x5d69d7b8) # Feature transformer hash
    self.write_feature_transformer(model.factorizer, model.input)
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

  def write_feature_transformer(self, factorizer, layer):
    # int16 bias = round(x * 127)
    # int16 weight = round(x * 127)
    bias = layer.bias.data
    bias = bias.mul(127).round().to(torch.int16)
    ascii_hist('ft bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())

    weight = layer.weight.data
    if factorizer is not None:
      weight = factorizer.coalesce_weights(weight)
    weight = weight.mul(127).round().to(torch.int16)
    ascii_hist('ft weight:', weight.numpy())
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
    ascii_hist('fc bias:', bias.numpy())
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
    ascii_hist('fc weight:', weight.numpy())
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int32(self, v):
    self.buf.extend(struct.pack("<i", v))

class NNUEReader():
  def __init__(self, f):
    self.f = f
    self.model = M.NNUE(factorizer=None)

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
    d = numpy.fromfile(self.f, dtype, reduce(operator.mul, shape, 1))
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

def test_eval_error(model, data_filename, num_samples):
  enable_factorizer = model.factorizer is not None
  features = halfkp
  features_name = features.Features.name
  if enable_factorizer:
    factorizer = features.Factorizer()
    features_name = factorizer.name
  batch_size = 1024
  num_samples = (num_samples + batch_size - 1) // batch_size * batch_size
  data_provider = nnue_dataset.SparseBatchDataset(features_name, data_filename, batch_size, num_workers=1, filtered=True)

  quantized_model = model.get_quantized()
  it = iter(data_provider)
  rel_errs = []
  avg_abs_evals = []
  avg_abs_evals_diff = []
  for i in range(num_samples // batch_size):
    batch = next(it)
    us, them, white, black, outcome, score = batch
    reference = model.forward(us, them, white, black) * 600.0
    quantized = quantized_model.forward(us, them, white, black)
    avg_abs_evals.append(reference.abs().mean())
    avg_abs_evals_diff.append((reference - quantized).abs().mean())
    rel_errs.append((reference - quantized).abs().mean() / reference.abs().mean())

  print('Avg abs eval:      {}'.format(sum(avg_abs_evals) / len(avg_abs_evals)))
  print('Avg abs eval diff: {}'.format(sum(avg_abs_evals_diff) / len(avg_abs_evals_diff)))
  print('Relative error:    {}'.format(sum(rel_errs) / len(rel_errs)))

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
  parser.add_argument("target", help="Target file (can be .pt or .nnue)")
  parser.add_argument("--test-eval-error", action='store_true', help="Measure eval error after quantization relative to before.")
  parser.add_argument("--test-data", type=str, help="The file to get data from. Use for example when measuring eval error.")
  parser.add_argument("--test-num-samples", type=int, help="The number of samples to use for any testing.")
  args = parser.parse_args()

  if args.test_eval_error:
    if not args.test_data:
      print('To test eval one must specify data file with --test-data.')
      raise Exception('Invalid parameters')

    if not args.test_num_samples or args.test_num_samples < 1:
      print('To test eval one must specify the number of samples to use with --test-num-samples')
      raise Exception('Invalid parameters')

  print('Converting %s to %s' % (args.source, args.target))

  if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
    if not args.target.endswith(".nnue"):
      raise Exception("Target file must end with .nnue")
    if args.source.endswith(".pt"):
      nnue = torch.load(args.source)
    else:
      # This try/catch is not ideal, but we don't know if we are converting
      # a factorized checkpoint, or an unfactorized one.
      try:
        # First try with defaults (factorizer on).
        nnue = M.NNUE.load_from_checkpoint(args.source)
      except:
        # If that fails, then try with factorizer disaabled.
        nnue = M.NNUE.load_from_checkpoint(args.source, factorizer=None)
    nnue.eval()
    if args.test_eval_error:
      test_eval_error(nnue, args.test_data, args.test_num_samples)
    writer = NNUEWriter(nnue)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  elif args.source.endswith(".nnue"):
    if not args.target.endswith(".pt"):
      raise Exception("Target file must end with .pt")
    with open(args.source, 'rb') as f:
      reader = NNUEReader(f)
    model = reader.model
    if args.test_eval_error:
      test_eval_error(model, args.test_data, args.test_num_samples)
    torch.save(model, args.target)
  else:
    raise Exception('Invalid filetypes: ' + str(args))

if __name__ == '__main__':
  main()
