import model as M
import nnue_bin_dataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

def main():
  nnue = M.NNUE()
  train_data = DataLoader(nnue_bin_dataset.NNUEBinData('d8_100000.bin'), batch_size=128, shuffle=True, num_workers=4)
  val_data = DataLoader(nnue_bin_dataset.NNUEBinData('d10_10000.bin'), batch_size=32)

  tb_logger = pl_loggers.TensorBoardLogger('logs/')
  trainer = pl.Trainer(logger=tb_logger)
  trainer.fit(nnue, train_data, val_data)

if __name__ == '__main__':
  main()
