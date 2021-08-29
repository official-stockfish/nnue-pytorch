#!/bin/bash

cd ..

echo "python3 delete_bad_nets.py ../nnue-pytorch-training/experiment_$1/"
python3 "delete_bad_nets.py" "../nnue-pytorch-training/experiment_$1/"
