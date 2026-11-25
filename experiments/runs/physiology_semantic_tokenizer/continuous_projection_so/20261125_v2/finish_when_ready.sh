#!/bin/bash
set -eu
root=experiments/runs/physiology_semantic_tokenizer/continuous_projection_so/20261125_v2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
for seed in 17 43; do
  while [ ! -f "$root/evaluation/formal_continuous_$seed/summary.json" ]; do
    systemctl --user is-active --quiet "continuous-so-score-$seed" || exit 1
    sleep 10
  done
done
.venv/bin/python -u experiments/scripts/summarize_continuous_so.py
.venv/bin/python -u experiments/scripts/render_continuous_so_review.py --output "$root/review_v1"
