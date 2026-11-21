#!/usr/bin/env bash
set -euo pipefail
out=experiments/runs/physiology_semantic_tokenizer/observation_factorization/20261120_hb_coordinate_sim_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
exec .venv/bin/python -u experiments/scripts/run_hb_word_coordinate_simulation.py --output "$out" >> "$out/process.log" 2>&1
