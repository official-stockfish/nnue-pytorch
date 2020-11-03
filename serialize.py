import model as M
import numpy
import nnue_bin_dataset
import struct
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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
    self.int32(0x7AF32F17) # version
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
    print(numpy.histogram(bias.numpy()))
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    weight = weight.mul(127).round().to(torch.int16)
    print(numpy.histogram(weight.numpy()))
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    self.buf.extend(weight.transpose(0, 1).flatten().numpy().tobytes())

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
    else:
      kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
    kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
    kMaxWeight = 127.0 / kWeightScale # roughly 2.0

    # int32 bias = round(x * kBiasScale)
    # int8 weight = round(x * kWeightScale)
    bias = layer.bias.data
    bias = bias.mul(kBiasScale).round().to(torch.int32)
    print(numpy.histogram(bias.numpy()))
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data
    weight = weight.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)
    print(weight.shape)
    print(numpy.histogram(weight.numpy()))
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int16(self, v):
    self.buf.extend(struct.pack("<h", v))

  def int32(self, v):
    self.buf.extend(struct.pack("<i", v))

class NNUEReader():
  def __init__(self, buf):
    self.buf = buf

def main():
  nnue = M.NNUE.load_from_checkpoint('last.ckpt')
  writer = NNUEWriter(nnue)
  with open('quantized.nnue', 'wb') as f:
    f.write(writer.buf)

if __name__ == '__main__':
  main()
