#!/bin/bash
set -eu
cd /home/uais7/pid-mcm-implementation
run=experiments/runs/physiology_semantic_tokenizer/all_channel_span/20261127_v1
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
stage=$1
seed=$2
workers=$3
exec .venv/bin/python experiments/scripts/run_all_channel_span.py "$stage" --seed "$seed" --workers "$workers" >> "$run/${stage}_${seed}.log" 2>&1
