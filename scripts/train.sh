#!/bin/bash

python train.py ../data/d5_2b.binpack ../data/d10_128000_6293.binpack --lambda 0.8 --gpus 1 --val_check_interval 2000 --threads 2 --batch-size 16384 --progress_bar_refresh_rate 20
