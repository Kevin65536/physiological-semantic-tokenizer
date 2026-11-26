#!/bin/bash
set -eu
cd /home/uais7/pid-mcm-implementation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python experiments/scripts/run_nback_function_screen.py extract --workers 10
.venv/bin/python experiments/scripts/run_nback_function_screen.py score
