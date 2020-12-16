#!/bin/bash

python train.py ../data/d1_1b.binpack ../data/d10_128000_6293.binpack --gpus 1 --val_check_interval 2000 --threads 2 --batch-size 16384 --progress_bar_refresh_rate 20 --smart-fen-skipping --random-fen-skipping 10
