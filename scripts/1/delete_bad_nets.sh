#!/bin/bash

# Usage:
#    ./delete_bad_nets.sh EXPERIMENT_ID [PRESERVE_N]
# Preserves PRESERVE_N best nets by ordo.out. Default: 16.

# Get the directory of the script file being executed. Resolves links.
BASEDIR=$(dirname $(readlink -f "$0"))

# Load the configuration file. This is not secure so make sure
# the configuration is trusted.
source "$BASEDIR/config.sh"

EXPERIMENT_ID="$1"
EXPERIMENT_DIR="$EXPERIMENTS_DIR/experiment_$EXPERIMENT_ID"
PRESERVE_N="${2:-16}"

echo "Executing..."
python3 "$BASEDIR/delete_bad_nets.py"
    "$EXPERIMENT_DIR"
    "$PRESERVE_N"
