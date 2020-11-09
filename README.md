# Setup
```
python3 -m venv env
source env/bin/activate
pip install pytorch-lightning
pip install python-chess
pip install tensorboard
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

# Export the network

Put the checkpoint you want to export in `last.ckpt`, then:
```
python serialize.py
```
It will write out `serialized.nnue`, which you can directly
load in Stockfish.

# Logging

```
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/

# Thanks

* Sopel - for the amazing fast sparse data loader
* syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
* https://github.com/connormcmonigle/seer-nnue
* https://github.com/DanielUranga/TensorFlowNNUE
* https://hxim.github.io/Stockfish-Evaluation-Guide/
