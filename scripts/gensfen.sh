#!/bin/bash

# Using this commit:
# https://github.com/Sopel97/Stockfish.git
# commit d7d4ec211f7ef35ff39fe8aea54623a468b36c7d

DEPTH=5
GAMES=128000000
SEED=$RANDOM
 
options="
uci
setoption name PruneAtShallowDepth value false
setoption name Use NNUE value pure
setoption name Threads value 250
setoption name Hash value 10240
setoption name SyzygyPath value /dev/shm/vjoost/3-4-5-6/WDL/:/dev/shm/vjoost/3-4-5-6/DTZ/
isready
gensfen set_recommended_uci_options ensure_quiet random_multi_pv 4 random_multi_pv_diff 50 random_move_count 8 random_move_maxply 20 write_minply 5 eval_limit 1000 seed $SEED depth $DEPTH loop $GAMES output_file_name d${DEPTH}_${GAMES}_${SEED}"
 
printf "$options" | ./stockfish
