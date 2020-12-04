# Setup
```
python3 -m venv env
source env/bin/activate
pip install python-chess==0.31.4 pytorch-lightning torch
```

# Build the fast DataLoader
This requires a C++17 compiler.

Windows:
```
compile_data_loader.bat
```

Linux/Mac:
```
sh compile_data_loader.bat
```

# Train a network

```
source env/bin/activate
python train.py train_data.bin val_data.bin
```

## Resuming from a checkpoint
```
python train.py --resume_from_checkpoint <path> ...
```

## Training on GPU
```
python train.py --gpus 1 ...
```

## Enable factorizer
```
python train.py --enable-factorizer ...
```

# Export a network

Using either a checkpoint (`.ckpt`) or serialized model (`.pt`),
you can export to SF NNUE format.  This will convert `last.ckpt`
to `nn.nnue`, which you can load directly in SF.
```
python serialize.py last.ckpt nn.nnue
```

# Import a network

Import an existing SF NNUE network to the pytorch network format.
```
python serialize.py nn.nnue converted.pt
```

# Logging

```
pip install tensorboard
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/

# Thanks

* Sopel - for the amazing fast sparse data loader
* connormcmonigle - https://github.com/connormcmonigle/seer-nnue, and loss function advice.
* syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
* https://github.com/DanielUranga/TensorFlowNNUE
* https://hxim.github.io/Stockfish-Evaluation-Guide/
* dkappe - Suggesting ranger (https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
