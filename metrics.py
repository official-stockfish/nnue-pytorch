import argparse
import nnue_bin_dataset
import torch
import model as M
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

def compute_mse(nnue, data):
  errors = []
  for i in range(0, len(data), 1000):
    raw = data.get_raw(i)
    board, move, turn, score = raw
    #cp =  M.cp_conversion(torch.tensor([score])).item()
    x = data[i]
    x = [v.reshape((1,-1)) for v in x]
    # Multiply by 600 to match scaling factor
    ev = nnue(x[0], x[1], x[2], x[3]).item() * 600
    print(board.fen())
    kPawnValueEg = 208.0
    print('dataset score:', score / kPawnValueEg, 'net:', ev / kPawnValueEg)
    errors.append((ev - score)**2)
  return sum(errors) / len(errors)

def main():
  parser = argparse.ArgumentParser(description="Runs evaluation for a model.")
  parser.add_argument("model", help="Source file (can be .ckpt, .pt or .nnue)")
  parser.add_argument("--dataset", default="d8_100000.bin", help="Dataset to evaluate on (.bin)")
  args = parser.parse_args()

  if args.model.endswith(".pt"):
    nnue = torch.load(args.model, map_location=torch.device('cpu'))
  else:
    nnue = M.NNUE.load_from_checkpoint(args.model)
  data = nnue_bin_dataset.NNUEBinData(args.dataset)

  #trainer = pl.Trainer()
  #trainer.test(nnue, DataLoader(data, batch_size=128))
  print('MSE:', compute_mse(nnue, data))

if __name__ == '__main__':
  main()
