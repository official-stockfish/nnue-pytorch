#!/bin/bash

# Usage:
#    ./train_continuous.sh [RUN_ID] [GPU_ID]
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
RUN_ID="${1:-0}"
GPU_ID="${2:-$RUN_ID}"
EXPERIMENT_DIR="$BASEDIR/.."
BASE_RUN_DIR="$EXPERIMENT_DIR/run_$RUN_ID"
ITERATION=0

if [ ! -f "$EXPERIMENT_DIR/.experiment" ]
then
    echo "Run setup_experiment.sh before running training."
    exit
fi

while true
do
    RUN_DIR="${BASE_RUN_DIR}_$ITERATION"

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

    ITERATION=$((ITERATION + 1))
done
