#!/bin/bash

# Usage:
#    ./delete_bad_nets.sh [PRESERVE_N]
# Preserves PRESERVE_N best nets by ordo.out. Default: 16.

# Get the directory of the script file being executed. Resolves links.
BASEDIR=$(dirname $(readlink -f "$0"))

# Load the configuration file. This is not secure so make sure
# the configuration is trusted.
source "$BASEDIR/config.sh"

EXPERIMENT_DIR="$BASEDIR/.."
PRESERVE_N="${1:-16}"

if [ ! -f "$EXPERIMENT_DIR/.experiment" ]
then
    echo "Run setup_experiment.sh before running training."
    exit
fi

echo "Executing..."
python3 "$BASEDIR/delete_bad_nets.py" \
    "$EXPERIMENT_DIR" \
    "$PRESERVE_N"
