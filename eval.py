import model as M
import nnue_bin_dataset
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

def main():
  nnue = M.NNUE.load_from_checkpoint('epoch=13.ckpt')
  data = nnue_bin_dataset.NNUEBinData('d8_100000.bin')

  errors = []
  for i in range(0, len(data), 1000):
    raw = data.get_raw(i)
    board, move, turn, score = raw
    cp =  M.cp_conversion(torch.tensor([score])).item()
    x = data[i]
    x = [v.reshape((1,-1)) for v in x[:3]]
    ev = nnue(x).item()
    print('dataset cp:',score / 100.0, 'score:', cp, 'net:', ev)
    errors.append((ev - cp)**2)
  print('MSE:', sum(errors) / len(errors))

if __name__ == '__main__':
  main()
