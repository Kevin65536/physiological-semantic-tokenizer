#!/bin/bash
set -eu
root=experiments/runs/physiology_semantic_tokenizer/continuous_projection_so/20261125_v2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
if [ ! -f "$root/prepared.json" ]; then .venv/bin/python -u experiments/scripts/train_continuous_so_formal.py prepare; fi
bash "$root/gpu.sh" 4B experiments/scripts/train_continuous_so_formal.py train --seed 17 > "$root/seed17.log" 2>&1 &
a=$!
bash "$root/gpu.sh" CA experiments/scripts/train_continuous_so_formal.py train --seed 43 > "$root/seed43.log" 2>&1 &
b=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
exit "$rc"
