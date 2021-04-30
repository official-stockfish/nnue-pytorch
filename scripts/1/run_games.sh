#!/bin/bash

# Usage:
#    ./run_games.sh

# Get the directory of the script file being executed. Resolves links.
BASEDIR=$(dirname $(readlink -f "$0"))

# Load the configuration file. This is not secure so make sure
# the configuration is trusted.
source "$BASEDIR/config.sh"

EXPERIMENT_ID="$1"
EXPERIMENT_DIR="$BASEDIR/.."
SF_TEST_PATH="$EXPERIMENT_DIR/stockfish_test/src/stockfish$EXE_SUFFIX"
SF_BASE_PATH="$EXPERIMENT_DIR/stockfish_base/src/stockfish$EXE_SUFFIX"

if [ ! -f "$EXPERIMENT_DIR/.experiment" ]
then
    echo "Run setup_experiment.sh before running training."
    exit
fi

echo "Starting tests..."
python3 "$BASEDIR/run_games.py" \
    --concurrency="$TEST_CONCURRENCY" \
    --ordo_exe "$ORDO_PATH" \
    --c_chess_exe "$C_CHESS_CLI_PATH" \
    --stockfish_base "$SF_BASE_PATH" \
    --stockfish_test "$SF_TEST_PATH" \
    --book_file_name "$OPENING_BOOK_PATH" \
    --tc "$TEST_TC" \
    --hash_size "$TEST_HASH" \
    --features "$FEATURE_SET" \
    --net_serializer "$EXPERIMENT_DIR/nnue-pytorch/serialize.py" \
    --explore_factor "$TEST_EXPLORE_FACTOR" \
    "$EXPERIMENT_DIR"
