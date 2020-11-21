import argparse
import model as M
import nnue_dataset
import nnue_bin_dataset
import pytorch_lightning as pl
import halfkp
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

def data_loader_cc(train_filename, val_filename, num_workers, batch_size, devices):
  # Epoch and validation sizes are arbitrary
  epoch_size = 100000000
  val_size = 1000000
  train_infinite = nnue_dataset.SparseBatchDataset(halfkp.FACTOR_NAME, train_filename, batch_size, num_workers=num_workers, devices=devices)
  val_infinite = nnue_dataset.SparseBatchDataset(halfkp.FACTOR_NAME, val_filename, batch_size, devices=devices)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def data_loader_py(train_filename, val_filename, batch_size):
  train = DataLoader(nnue_bin_dataset.NNUEBinData(train_filename), batch_size=batch_size, shuffle=True, num_workers=4)
  val = DataLoader(nnue_bin_dataset.NNUEBinData(val_filename), batch_size=32)
  return train, val

def main():
  parser = argparse.ArgumentParser(description="Trains the network.")
  parser.add_argument("train", help="Training data (.bin or .binpack)")
  parser.add_argument("val", help="Validation data (.bin or .binpack)")
  parser = pl.Trainer.add_argparse_args(parser)
  parser.add_argument("--py-data", action="store_true", help="Use python data loader (default=False)")
  parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
  parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Number of worker threads to use for data loading. Currently only works well for binpack.")
  parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
  parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
  parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
  args = parser.parse_args()

  devices = ['cuda:0']

  nnue = M.NNUE(halfkp, lambda_=args.lambda_, devices=devices)

  print("Training with {} validating with {}".format(args.train, args.val))

  pl.seed_everything(args.seed)
  print("Seed {}".format(args.seed))

  batch_size = args.batch_size
  if batch_size <= 0:
    batch_size = 128 if args.gpus == 0 else 8192

  # Not sure where this should be done.
  # We need to reduce the batch size by the number of used devices
  # because we do one batch per device; the data loader produces
  # N batches when there's N devices.
  batch_size //= len(devices)

  print('Using batch size {}'.format(batch_size))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  if args.py_data:
    print('Using python data loader')
    train, val = data_loader_py(args.train, args.val, batch_size)
  else:
    print('Using c++ data loader')
    train, val = data_loader_cc(args.train, args.val, args.num_workers, batch_size, devices=devices)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)

  tb_logger = pl_loggers.TensorBoardLogger(logdir)
  trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger)
  trainer.fit(nnue, train, val)

if __name__ == '__main__':
  main()
