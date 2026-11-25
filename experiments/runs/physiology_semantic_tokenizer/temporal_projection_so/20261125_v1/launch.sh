#!/usr/bin/env bash
set -eu
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec .venv/bin/python -u experiments/scripts/evaluate_temporal_projection_so.py --workers 8
