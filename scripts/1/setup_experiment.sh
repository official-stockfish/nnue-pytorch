#!/bin/bash

# Usage:
#    ./setup_experiment.sh EXPERIMENT_ID

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
EXPERIMENT_DIR="$EXPERIMENTS_DIR/experiment_$EXPERIMENT_ID"

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
    mkdir "$EXPERIMENT_DIR/stockfish_test"
    mkdir "$EXPERIMENT_DIR/stockfish_base"

    echo "Copying scripts..."
    cp -R "$BASEDIR/." "$EXPERIMENT_DIR/scripts/"

    echo "Copying the trainer..."
    cp -R "$NNUE_PYTORCH_DIR/." "$EXPERIMENT_DIR/nnue-pytorch/"

    if $INCLUDE_STOCKFISH
    then
        echo "Copying test stockfish..."
        cp -R "$STOCKFISH_TEST_DIR/." "$EXPERIMENT_DIR/stockfish_test/"

        echo "Copying base stockfish..."
        cp -R "$STOCKFISH_BASE_DIR/." "$EXPERIMENT_DIR/stockfish_base/"

        # Always compile the data loader and stockfish after copying
        # to ensure the correct versions
        echo "Compiling test stockfish..."
        if (cd "$EXPERIMENT_DIR/stockfish_test/src/" && make "build" "ARCH=$ARCH" "-j")
        then
            echo "Stockfish test compilation successful."
        else
            echo "Stockfish test compilation failed."
            exit
        fi

        echo "Compiling base stockfish..."
        if (cd "$EXPERIMENT_DIR/stockfish_base/src/" && make "build" "ARCH=$ARCH" "-j")
        then
            echo "Stockfish base compilation successful."
        else
            echo "Stockfish base compilation failed."
            exit
        fi
    fi

    echo "Compiling the data loader..."
    if (cd "$EXPERIMENT_DIR/nnue-pytorch/" && "./compile_data_loader.bat")
    then
        echo "Data loader compilation successful."
    else
        echo "Data loader compilation failed."
        exit
    fi

    # Create a file indicating that this is an experiment
    # directory and that it was setup correctly.
    touch "$EXPERIMENT_DIR/.experiment"
else
    echo "Experiment already set up."
fi
