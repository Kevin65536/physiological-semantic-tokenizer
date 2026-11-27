#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. PYTHONUNBUFFERED=1
dest=experiments/runs/physiology_semantic_tokenizer/typed_span/20261127_v1
seed="$2"
while systemctl --user is-active --quiet "typed-span-fit$seed.service"; do sleep 2; done
for arm in joint no_lag ss_only no_anchor frozen; do
    test -s "$dest/fits/${arm}_${seed}/done.json"
done
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -v bus="$1" '$1 == bus"," {print $2}')"
for arm in joint no_lag ss_only no_anchor frozen; do
    .venv/bin/python experiments/scripts/run_typed_span.py decode --arm "$arm" --seed "$seed" >> "$dest/decode_${seed}.log" 2>&1
done
