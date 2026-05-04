#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

python -u build_compatible_engine.py --engine-dest-path /data/tmp/stockfish --overwrite
python -u ../../cross_check_eval.py --net /data/logs/training/runs/unittests_train_pipeline_cpu/lightning_logs/version_2/checkpoints/last.nnue --data .pgo/small.binpack --engine /data/tmp/stockfish --build_engine_from_sha='' --device='cpu'
