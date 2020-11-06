import model as M
import nnue_bin_dataset
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

def compute_mse(nnue, data):
  errors = []
  for i in range(0, len(data), 1000):
    raw = data.get_raw(i)
    board, move, turn, score = raw
    cp =  M.cp_conversion(torch.tensor([score])).item()
    x = data[i]
    x = [v.reshape((1,-1)) for v in x]
    ev = nnue(x[0], x[1], x[2], x[3]).item()
    #print('dataset cp:', score / 100.0, 'score:', cp, 'net:', ev)
    errors.append((ev - cp)**2)
  return sum(errors) / len(errors)

def main():
  nnue = M.NNUE.load_from_checkpoint('last.ckpt')
  data = nnue_bin_dataset.NNUEBinData('d8_100000.bin')

  print('MSE:', compute_mse(nnue, data))

if __name__ == '__main__':
  main()
