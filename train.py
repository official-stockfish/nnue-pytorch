import model as M
import halfkp_dataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

def main():
  nnue = M.NNUE()
  #train_data = DataLoader(nnue_bin_dataset.NNUEBinData('d8_100000.bin'), batch_size=128, shuffle=True, num_workers=1)

  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train_data = DataLoader(halfkp_dataset.SparseBatchDataset('d8_100000.bin', 8192), batch_size=None, batch_sampler=None)
  val_data = DataLoader(halfkp_dataset.SparseBatchDataset('d10_10000.bin', 1024), batch_size=None, batch_sampler=None)
  tb_logger = pl_loggers.TensorBoardLogger('logs/')
  trainer = pl.Trainer(logger=tb_logger, gpus=1)
  trainer.fit(nnue, train_data, val_data)

if __name__ == '__main__':
  main()
