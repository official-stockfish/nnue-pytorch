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

Filenames are hardcoded right now, edit train.py first, then:
```
source env/bin/activate
python train.py
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
* syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
* https://github.com/connormcmonigle/seer-nnue
* https://github.com/DanielUranga/TensorFlowNNUE
* https://hxim.github.io/Stockfish-Evaluation-Guide/
