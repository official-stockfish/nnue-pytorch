#!/bin/bash

# Usage:
#    ./run_games.sh EXPERIMENT_ID

# Get the directory of the script file being executed. Resolves links.
BASEDIR=$(dirname $(readlink -f "$0"))

# Load the configuration file. This is not secure so make sure
# the configuration is trusted.
source "$BASEDIR/config.sh"

EXPERIMENT_ID="$1"
EXPERIMENT_DIR="$EXPERIMENTS_DIR/experiment_$EXPERIMENT_ID"
COPIED_SF_BASE_PATH="$EXPERIMENT_DIR/stockfish_base$EXE_SUFFIX"
SF_TEST_PATH="$EXPERIMENT_DIR/Stockfish/src/stockfish$EXE_SUFFIX"

# We always preserve the base stockfish so that the test is consistent
# and the experiment is self contained.
echo "Copying the base stockfish..."
cp "$TEST_SF_BASE_PATH" "$COPIED_SF_BASE_PATH"

echo "Starting tests..."
python3 "$BASEDIR/run_games.py" \
    --concurrency="$TEST_CONCURRENCY" \
    --ordo_exe "$ORDO_PATH" \
    --c_chess_exe "$C_CHESS_CLI_PATH" \
    --stockfish_base "$COPIED_SF_BASE_PATH" \
    --stockfish_test "$SF_TEST_PATH" \
    --book_file_name "$OPENING_BOOK_PATH" \
    --tc "$TEST_TC" \
    --hash_size "$TEST_HASH" \
    --features "$FEATURE_SET" \
    --net_serializer "$EXPERIMENT_DIR/nnue-pytorch/serialize.py" \
    --explore_factor "$TEST_EXPLORE_FACTOR" \
    "$EXPERIMENT_DIR"
