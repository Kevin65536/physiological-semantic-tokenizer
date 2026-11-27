#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
dest=experiments/runs/physiology_semantic_tokenizer/typed_span_expanded/20261127_v2
stage=$1
if [[ "$stage" == "prep_pilot" || "$stage" == "prepare" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  .venv/bin/python experiments/scripts/run_expanded_typed_span.py "$stage" --workers "$2" >> "$dest/${stage}.log" 2>&1
else
  bus=$2
  [[ "$bus" != "00000000:17:00.0" ]]
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -v bus="$bus" '$1 == bus"," {print $2}')"
  seed=$3
  shift 3
  for arm in "$@"; do
    .venv/bin/python experiments/scripts/run_expanded_typed_span.py "$stage" --arm "$arm" --seed "$seed" >> "$dest/${stage}_${arm}_${seed}.log" 2>&1
  done
fi
