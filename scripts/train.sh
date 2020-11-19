#!/bin/bash

python train.py ../data/d5_1b.binpack ../data/d10_128000_6293.binpack --lambda 0.8 --gpus 1 --val_check_interval 10000
