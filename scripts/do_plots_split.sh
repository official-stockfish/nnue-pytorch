#!/bin/bash

cd ..

out_file="../nnue-pytorch-training/plots/plot"
root_dirs=()
for arg in "$@"; do
    root_dirs+=("../nnue-pytorch-training/experiment_$arg")
    out_file+="_$arg"
done
out_file+="_split.png"

echo "python3 do_plots.py $out_file ${root_dirs[@]}"
python3 "do_plots.py" "--split" "--output" "$out_file" "${root_dirs[@]}"
