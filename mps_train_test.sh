#!/usr/bin/env bash
set -euo pipefail

# Allow MPS to use all available unified memory instead of the default 50% limit.
# Without this, the threats net optimizer states (~5 GB) can silently stall MPS.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python3 train.py \
  test79-2022-05-may-12tb7p.min-v2.binpack \
  --accelerator=mps \
  --compile-backend=none \
  --batch-size=65536 \
  --max_epochs=800 \
  --features=Full_Threats+HalfKAv2_hm^ \
  --l1=1024 \
  --l2=31 \
  --lr=4.375e-4 \
  --gamma=0.995 \
  --start-lambda=1.0 \
  --end-lambda=0.75 \
  --random-fen-skipping=10 \
  --early-fen-skipping=12 \
  --pc-y1=0.6893201149773951 \
  --pc-y2=2.9285769485515805 \
  --pc-y3=1.4386005301749225 \
  --w1=3.3553547771220007 \
  --w2=0.7006821612968052 \
  --num-workers=1 \
  --threads=1
