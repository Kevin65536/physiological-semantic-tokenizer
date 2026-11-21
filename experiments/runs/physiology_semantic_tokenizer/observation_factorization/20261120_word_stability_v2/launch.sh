#!/usr/bin/env bash
set -euo pipefail
out=experiments/runs/physiology_semantic_tokenizer/observation_factorization/20261120_word_stability_v2
exec >> "$out/process.log" 2>&1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
if [[ ! -f "$out/preflight.json" ]]; then
    .venv/bin/python -u experiments/scripts/run_observation_word_stability.py --stage preflight --output "$out"
fi
exec .venv/bin/python -u experiments/scripts/run_observation_word_stability.py --stage screen --output "$out"
