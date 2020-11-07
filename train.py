import model as M
import nnue_bin_dataset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

def main():
  nnue = M.NNUE()
  #train_data = DataLoader(nnue_bin_dataset.NNUEBinData('d8_100000.bin'), batch_size=128, shuffle=True, num_workers=1)

  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  # I expect that if batching is done on the C++ side and here
  # we just copy to tensors it would be fast enough
  # If though it turns out that workers are needed then it could be done one worker per file
  # It's also possible (and quite easy to do) to prepare the data on the C++ side
  # in a dedicated thread and then when calling get_next_entry_halfkp_dense
  # it would return already prepared entries/batches
  #train_data = DataLoader(nnue_bin_dataset.NNUEExternalDataDense('d8_100000.bin'), batch_size=128, num_workers=1)
  #train_data = DataLoader(nnue_bin_dataset.NNUEExternalDataDenseBatch('d8_100000.bin', 128), batch_size=None, batch_sampler=None, num_workers=1)
  train_data = DataLoader(nnue_bin_dataset.NNUEExternalDataSparseBatch('d8_100000.bin', 8192), batch_size=None, batch_sampler=None)
  #val_data = DataLoader(nnue_bin_dataset.NNUEBinData('d10_10000.bin'), batch_size=32)
  val_data = DataLoader(nnue_bin_dataset.NNUEExternalDataSparseBatch('d10_10000.bin', 32), batch_size=None, batch_sampler=None)
  tb_logger = pl_loggers.TensorBoardLogger('logs/')
  trainer = pl.Trainer(logger=tb_logger, gpus=1)
  trainer.fit(nnue, train_data, val_data)

if __name__ == '__main__':
  main()
