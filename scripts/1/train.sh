#!/bin/bash

# Usage:
#    ./train.sh EXPERIMENT_ID [RUN_ID] [GPU_ID]
# if GPU_ID is not specified it will be the same as RUN_ID
# if RUN_ID is not specified it is assumed to be 0

# Get the directory of the script file being executed. Resolves links.
BASEDIR=$(dirname $(readlink -f "$0"))

# Load the configuration file. This is not secure so make sure
# the configuration is trusted.
source "$BASEDIR/config.sh"

# All runs from the same experiment will be put into one directory.
# Separate runs into separate directories within.
# Each run contains the pytorch logs, checkpoints, and when games are ran
# it will also contain the respective .nnue files, ordo.out, and out.pgn.
EXPERIMENT_ID="$1"
RUN_ID="${2:-0}"
GPU_ID="${3:-$RUN_ID}"
EXPERIMENT_DIR="$EXPERIMENTS_DIR/experiment_$EXPERIMENT_ID"
RUN_DIR="$EXPERIMENT_DIR/run_$RUN_ID"

if [ ! -d "$EXPERIMENTS_DIR" ]
then
    mkdir "$EXPERIMENTS_DIR"
fi

if [ ! -d "$EXPERIMENT_DIR" ]
then
    # We prepare the directory layout.
    # Each experiment directory contains the copies of the
    # pytorch trainer and the version of Stockfish that will
    # use these nets. This is to ensure the setup is consistent
    # and persistent and the original copy can be modified during the run.
    echo "Experiment does not yet exists. Creating directory structure..."
    mkdir "$EXPERIMENT_DIR"
    mkdir "$EXPERIMENT_DIR/nnue-pytorch"
    mkdir "$EXPERIMENT_DIR/Stockfish"

    echo "Copying the trainer..."
    cp -R "$NNUE_PYTORCH_DIR/." "$EXPERIMENT_DIR/nnue-pytorch/"

    echo "Copying stockfish..."
    cp -R "$STOCKFISH_DIR/." "$EXPERIMENT_DIR/Stockfish/"

    echo "Copying scripts..."
    cp -R "$BASEDIR/." "$EXPERIMENT_DIR/scripts/"

    # Always compile the data loader and stockfish after copying
    # to ensure the correct versions
    echo "Compiling stockfish..."
    if (cd "$EXPERIMENT_DIR/Stockfish/src/" && make "build" "ARCH=$ARCH" "-j")
    then
        echo "Stockfish compilation successful."
    else
        echo "Stockfish compilation failed."
        exit
    fi

    echo "Compiling the data loader..."
    if (cd "$EXPERIMENT_DIR/nnue-pytorch/" && "./compile_data_loader.bat")
    then
        echo "Data loader compilation successful."
    else
        echo "Data loader compilation failed."
        exit
    fi
else
    echo "Experiment already set up."
fi

if [ -d "$RUN_DIR" ]
then
    echo "Run already exists. Choose a different run number."
    exit
else
    mkdir "$RUN_DIR"
fi

echo "Starting the training..."
# It's important that pwd is the directory with the script because it affects
# the way dynamic libraries are searched. Otherwise it doesn't find the data loader.
(cd "$EXPERIMENT_DIR/nnue-pytorch/" && \
    python3 "train.py" \
        "$TRAINING_DATA_PATH" \
        "$VALIDATION_DATA_PATH" \
        --gpus "$GPU_ID," \
        --threads "$NUM_TORCH_THREADS" \
        --num-workers "$NUM_DATALOADER_THREADS" \
        --batch-size "$BATCH_SIZE" \
        --progress_bar_refresh_rate "$PROGRESS_BAR_REFRESH_RATE" \
        --smart-fen-skipping \
        --random-fen-skipping "$RANDOM_FEN_SKIPPING" \
        --features "$FEATURE_SET" \
        --lambda "$LAMBDA" \
        --max_epochs "$MAX_EPOCHS" \
        --default_root_dir "$RUN_DIR" \
        --ckpt-save-policy "$CKPT_SAVE_POLICY" \
        --ckpt-save-period "$CKPT_SAVE_PERIOD")
