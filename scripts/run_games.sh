#!/bin/bash

cd ..

python3 run_games.py \
    --concurrency=24 \
    --ordo_exe ordo \
    --c_chess_exe c-chess-cli \
    --features=HalfKAv2_hm^ \
    --stockfish_base ../nnue-pytorch-training/stockfish/stockfish \
    --stockfish_test ../nnue-pytorch-training/experiment_$1/stockfish \
    --book_file_name ../nnue-pytorch-training/book/noob_3moves.epd \
    ../nnue-pytorch-training/experiment_$1/
