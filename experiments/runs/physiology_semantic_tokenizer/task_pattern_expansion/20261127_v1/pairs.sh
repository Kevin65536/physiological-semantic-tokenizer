#!/bin/bash
set -eu
cd /home/uais7/pid-mcm-implementation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while systemctl --user is-active --quiet task-pattern-expansion-1127-extract; do sleep 2; done
.venv/bin/python experiments/scripts/score_task_pattern_pairing.py
