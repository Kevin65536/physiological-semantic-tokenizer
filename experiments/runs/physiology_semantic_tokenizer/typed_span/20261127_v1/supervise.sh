#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. PYTHONUNBUFFERED=1
dest=experiments/runs/physiology_semantic_tokenizer/typed_span/20261127_v1
gpu=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -v bus="$1" '$1 == bus"," {print $2}')
[[ "$1" != 00000000:17:00.0 && "$gpu" == GPU-* ]]
export CUDA_VISIBLE_DEVICES="$gpu"
seed="$2"
shift 2
for arm in "$@"; do
    stage=run
    if [[ "$arm" == pilot ]]; then stage=pilot; arm=joint; fi
    .venv/bin/python experiments/scripts/run_typed_span.py "$stage" --arm "$arm" --seed "$seed" >> "$dest/${stage}_${arm}_${seed}.log" 2>&1
done
