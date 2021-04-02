# Set of scripts for NNUE training.

This set of scripts is meant to work on linux, but it *might* also work on windows with MSYS2.

The basic principles are the following:

- each experiment is self contained
    - contains the code of the trainer used
    - contains the code of the player used
- this script directory is self contained
    - it doesn't require anything from outside to be under a specific path
- only copied files are executed
    - after the script has set up an experiment the base source directories can change
- a single configuration file (`config.sh`)

The configuration is fairly straightforward. All parameters are explained in the `config.sh` file. Once the configuration is personalized one needs to change `APPROVED` to a value other than 0, otherwise trying to load te configuration will fail.

An experiment has the following directory structure
```
- $EXPERIMENTS_DIR
    - experiment_$EXP_ID
        - stockfish_base[.exe]
            the executable for the baseline
        - run_$RUN_ID0
        - run_$RUN_ID...
            - nn-epoch*.nnue
            - default
                - version_0
                    - tf events file
                    - checkpoints
                        - nn-epoch*.ckpt
                        - last.ckpt
        - nnue-pytorch
            copy of the trainer repo at the time of starting the experiment
        - Stockfish
            copy of the stockfish repo at the time of starting the experiment
            - src
                - stockfish[.exe]
                    the executable matching the trainer
```

With `config.sh` correctly configured the usual way to proceed is the following:

1. Run `./train.sh EXP_ID [RUN_ID] [GPU_ID]`, where `EXP_ID` is the experiment id, `RUN_ID` is the run id (default: 0), `GPU_ID` is the id of the gpu to run the training on (default: `RUN_ID`). You can train multiple runs of the same experiment in parallel (but don't start them at the exact same time. Before running the second run on the same experiment make sure the directory structure is setup and the training of the first run started, otherwise there are filesystem races that might break stuff).
2. Optionally run `./run_games.sh EXP_ID`. This will periodically check for new `.ckpt` files produced by all runs in the given experiment, convert them to `.nnue`, and will continously run games to determine the best nets. This can (and should) be running parallel to the training.

There are also other utilities.

`./do_plots.sh EXPERIMENT_ID [EXPERIMENT_ID2] [EXPERIMENT_ID3] ...` will produce plots of train loss, validation loss, and elo (if `ordo.out` is present in the experiment directory) for the given experiments (one plot). The plot is saved under a name containing all experiment ids in the configured plots directory.

`./delete_bad_nets.sh EXPERIMENT_ID [PRESERVE_N]` will delete the worst (as determined from `ordo.out`) nets (both `.nnue` and `.ckpt`) from the given experiment. Only the best `PRESERVE_N` nets will be kept (default: 16). This is a nice automated way of freeing some disk space.
