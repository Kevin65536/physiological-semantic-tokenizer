#!/bin/bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
exec .venv/bin/python -u experiments/scripts/check_continuous_so_targets.py
