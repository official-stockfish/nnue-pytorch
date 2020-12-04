#!/bin/bash

python train.py d8_128000_21865.binpack d8_128000_21865.binpack --lambda 1.0 --val_check_interval 2000 --threads 2 --batch-size 16384 --progress_bar_refresh_rate 20 --factorizer
