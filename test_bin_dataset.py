import torch
import nnue_bin_dataset

d = nnue_bin_dataset.NNUEBinData('d10_10000.bin')

for i in d:
  print(i)
