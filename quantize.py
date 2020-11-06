import model as M
import nnue_bin_dataset
import numpy
import pytorch_lightning as pl
import struct
import torch
from metrics import compute_mse
from torch.utils.data import DataLoader

class NNUEQuantizedWriter():
  """
  All values are stored in little endian.
  """
  def __init__(self, model):
    self.buf = bytearray()

    for m in model.modules():
      print(m)

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
    # int16 bias
    # int16 weight
    bias = layer.bias().data
    self.buf.extend(bias.flatten().numpy().astype(numpy.int16).tobytes())
    weight = layer.weight().data
    # weights stored as [41024][256], so we need to transpose the pytorch [256][41024]
    self.buf.extend(weight.transpose(0, 1).flatten().numpy().astype(numpy.int16).tobytes())

  def write_fc_layer(self, layer, is_output=False):
    # FC layers are stored as int8 weights, and int32 biases
    bias = layer.bias().data
    self.buf.extend(bias.flatten().numpy().astype(numpy.int32).tobytes())
    weight = layer.weight().data
    # Stored as [outputs][inputs], so we can flatten
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int16(self, v):
    self.buf.extend(struct.pack("<h", v))

  def int32(self, v):
    self.buf.extend(struct.pack("<i", v))

def main():
  nnue = M.NNUE.load_from_checkpoint('last.ckpt')
  nnue.eval()
  fuse_layers = [
    ['input', 'input_act'],
    ['l1', 'l1_act'],
    ['l2', 'l2_act'],
  ]
  torch.quantization.fuse_modules(nnue, fuse_layers, inplace=True)

  train = nnue_bin_dataset.NNUEBinData('d8_100000.bin')
  train_small = torch.utils.data.Subset(train, range(0, len(train) // 1000))
  train_loader = DataLoader(train_small)
  val_loader = DataLoader(nnue_bin_dataset.NNUEBinData('d10_10000.bin'))
  trainer = pl.Trainer()

  nnue.qconfig = torch.quantization.get_default_qconfig('fbgemm')
  nnue_prep = torch.quantization.prepare(nnue)
  trainer.test(nnue_prep, train_loader)
  nnue_int8 = torch.quantization.convert(nnue_prep)
  #trainer.test(nnue_int8, train_loader)

  #print('Baseline MSE:', compute_mse(nnue, train))
  #print('Quantized MSE:', compute_mse(nnue_int8, train))

  writer = NNUEQuantizedWriter(nnue_int8)
  with open('quantized.nnue', 'wb') as f:
    f.write(writer.buf)

  #torch.jit.save(torch.jit.script(nnue_int8), 'quantized.pt')
  #nnueq = torch.jit.load('quantized.pt')

if __name__ == '__main__':
  main()
