#!/bin/bash

cd ..

if [ ! -d "../nnue-pytorch-training/experiment_$1" ]
then
    mkdir ../nnue-pytorch-training/experiment_$1
    mkdir ../nnue-pytorch-training/experiment_$1/nnue-pytorch

    cp -R . ../nnue-pytorch-training/experiment_$1/nnue-pytorch/
fi

mkdir ../nnue-pytorch-training/experiment_$1/run_$2

python3 train.py \
    ../nnue-pytorch-training/data/large_gensfen_multipvdiff_100_d9.binpack \
    ../nnue-pytorch-training/data/large_gensfen_multipvdiff_100_d9.binpack \
    --gpus "$3," \
    --threads 1 \
    --num-workers 1 \
    --batch-size 16384 \
    --progress_bar_refresh_rate 20 \
    --random-fen-skipping 3 \
    --features=HalfKAv2_hm^ \
    --lambda=1.0 \
    --max_epochs=600 \
    --default_root_dir ../nnue-pytorch-training/experiment_$1/run_$2
