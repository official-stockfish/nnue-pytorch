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

def qmodel_from_model(ckpt_name):
  # Hardcoded to convert from HalfKP^ right now...
  factorized = features.get_feature_set_from_name('HalfKP^')
  #halfkp = features.get_feature_set_from_name('HalfKP')
  nnue = model.NNUE.load_from_checkpoint(ckpt_name, map_location='cpu', feature_set=factorized)
  nnue.eval()
  return nnue
  #qm = M.NNUE(halfkp)
  qm = M.NNUE(factorized)
  qm.eval()
  layers = ['input', 'l1', 'l2', 'l3', 'output']
  for name in layers:
    setattr(qm, name, getattr(nnue, name))
  #qm.input.weight = nn.Parameter(coalesce_ft_weights(nnue, nnue.input))
  #qm.input.in_features = halfkp.num_real_features
  return qm

def main():
  nnue = qmodel_from_model('epoch279_3layer.ckpt')

  halfkp = features.get_feature_set_from_name('HalfKP^')
  train = nnue_bin_dataset.NNUEBinData('large_gensfen_multipvdiff_100_d9.bin', halfkp)
  train_loader = DataLoader(torch.utils.data.Subset(train, range(0, 1000)))
  trainer = pl.Trainer()
  trainer.test(nnue, train_loader)

  fuse_layers = [
    ['input', 'input_act'],
    ['l1', 'l1_act'],
    ['l2', 'l2_act'],
  ]
  torch.quantization.fuse_modules(nnue, fuse_layers, inplace=True)

  nnue.qconfig = torch.quantization.get_default_qconfig('fbgemm')
  nnue_prep = torch.quantization.prepare(nnue)
  trainer.test(nnue_prep, train_loader)
  nnue_int8 = torch.quantization.convert(nnue_prep)
  trainer.test(nnue_int8, train_loader)

  #print('Baseline MSE:', compute_mse(nnue, train))
  #print('Quantized MSE:', compute_mse(nnue_int8, train))

  #torch.jit.save(torch.jit.script(nnue_int8), 'quantized.pt')
  #nnueq = torch.jit.load('quantized.pt')

if __name__ == '__main__':
  main()
