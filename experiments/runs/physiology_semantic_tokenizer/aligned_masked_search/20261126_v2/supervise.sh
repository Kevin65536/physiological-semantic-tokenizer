#!/bin/bash
set -euo pipefail
D=experiments/runs/physiology_semantic_tokenizer/aligned_masked_search/20261126_v2
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
exec .venv/bin/python -u experiments/scripts/run_aligned_masked_search.py controller >>"$D/controller.log" 2>&1
