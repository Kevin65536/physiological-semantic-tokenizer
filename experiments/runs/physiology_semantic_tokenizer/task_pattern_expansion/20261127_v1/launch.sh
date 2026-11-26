#!/bin/bash
set -eu
cd /home/uais7/pid-mcm-implementation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python experiments/scripts/run_task_pattern_expansion.py run --workers 12
