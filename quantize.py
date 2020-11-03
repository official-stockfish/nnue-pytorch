import model as M
import nnue_bin_dataset
import pytorch_lightning as pl
import torch
from metrics import compute_mse
from torch.utils.data import DataLoader

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
  trainer.test(nnue_int8, train_loader)

  print('Baseline MSE:')
  compute_mse(nnue, train)
  print('Quantized MSE:')
  compute_mse(nnue_int8, train)

  torch.jit.save(torch.jit.script(nnue_int8), 'quantized.pt')
  #nnueq = torch.jit.load('quantized.pt')

if __name__ == '__main__':
  main()
