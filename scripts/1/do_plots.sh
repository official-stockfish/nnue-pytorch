#!/bin/bash

# Usage:
#    ./do_plots.sh EXPERIMENT_ID [EXPERIMENT_ID2] [EXPERIMENT_ID3] ...

# Get the directory of the script file being executed. Resolves links.
BASEDIR=$(dirname $(readlink -f "$0"))

# Load the configuration file. This is not secure so make sure
# the configuration is trusted.
source "$BASEDIR/config.sh"

if [ ! -d "$PLOTS_DIR" ]
then
    mkdir "$PLOTS_DIR"
fi

# The resulting file name contains all the experiment ids.
OUT_FILE="$PLOTS_DIR/plot"
EXPERIMENT_DIRS=()
for arg in "$@"; do
    EXPERIMENT_DIRS+=("$EXPERIMENTS_DIR/experiment_$arg")
    OUT_FILE+="_$arg"
done
OUT_FILE+=".png"

echo "Plotting..."
python3 "$BASEDIR/do_plots.py" \
    "$OUT_FILE" \
    "${EXPERIMENT_DIRS[@]}"
