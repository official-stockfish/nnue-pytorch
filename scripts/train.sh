#!/bin/bash

python train.py \
 ../data/large_gensfen_multipvdiff_100_d9.binpack \
 ../data/large_gensfen_multipvdiff_100_d9.binpack \
 --gpus 1 \
 --val_check_interval 2000 \
 --threads 2 \
 --batch-size 8096 \
 --progress_bar_refresh_rate 20 \
 --smart-fen-skipping \
 --random-fen-skipping 13 \
 --features=HalfKP^ \
 --lambda=1.0 \
 --max_epochs=150
