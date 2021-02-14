import argparse
import model as M
import nnue_dataset
import nnue_bin_dataset
import pytorch_lightning as pl
import features
import os
import torch
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

def data_loader_cc(train_filename, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping, main_device):
  # Epoch and validation sizes are arbitrary
  epoch_size = 100000000
  val_size = 1000000
  features_name = feature_set.name
  train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def data_loader_py(train_filename, val_filename, feature_set, batch_size, main_device):
  train = DataLoader(nnue_bin_dataset.NNUEBinData(train_filename, feature_set), batch_size=batch_size, shuffle=True, num_workers=4)
  val = DataLoader(nnue_bin_dataset.NNUEBinData(val_filename, feature_set), batch_size=32)
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
  parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping', help="If enabled positions that are bad training targets will be skipped during loading. Default: False")
  parser.add_argument("--random-fen-skipping", default=0, type=int, dest='random_fen_skipping', help="skip fens randomly on average random_fen_skipping before using one.")
  parser.add_argument("--resume-from-model", dest='resume_from_model', help="Initializes training using the weights from the given .pt model")
  features.add_argparse_args(parser)
  args = parser.parse_args()

  if not os.path.exists(args.train):
    raise Exception('{0} does not exist'.format(args.train))
  if not os.path.exists(args.val):
    raise Exception('{0} does not exist'.format(args.val))

  feature_set = features.get_feature_set_from_name(args.features)

  if args.resume_from_model is None:
    nnue = M.NNUE(feature_set=feature_set, lambda_=args.lambda_)
  else:
    nnue = torch.load(args.resume_from_model)
    nnue.set_feature_set(feature_set)
    nnue.lambda_ = args.lambda_

  print("Feature set: {}".format(feature_set.name))
  print("Num real features: {}".format(feature_set.num_real_features))
  print("Num virtual features: {}".format(feature_set.num_virtual_features))
  print("Num features: {}".format(feature_set.num_features))

  print("Training with {} validating with {}".format(args.train, args.val))

  pl.seed_everything(args.seed)
  print("Seed {}".format(args.seed))

  batch_size = args.batch_size
  if batch_size <= 0:
    batch_size = 128 if args.gpus == 0 else 8192
  print('Using batch size {}'.format(batch_size))

  print('Smart fen skipping: {}'.format(args.smart_fen_skipping))
  print('Random fen skipping: {}'.format(args.random_fen_skipping))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)

  tb_logger = pl_loggers.TensorBoardLogger(logdir)
  checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True)
  trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=tb_logger)

  main_device = trainer.root_device if trainer.root_gpu is None else 'cuda:' + str(trainer.root_gpu)

  if args.py_data:
    print('Using python data loader')
    train, val = data_loader_py(args.train, args.val, feature_set, batch_size, main_device)
  else:
    print('Using c++ data loader')
    train, val = data_loader_cc(args.train, args.val, feature_set, args.num_workers, batch_size, args.smart_fen_skipping, args.random_fen_skipping, main_device)

  trainer.fit(nnue, train, val)

if __name__ == '__main__':
  main()
