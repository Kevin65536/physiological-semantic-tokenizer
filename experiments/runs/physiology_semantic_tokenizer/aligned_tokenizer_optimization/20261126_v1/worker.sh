#!/usr/bin/env bash
set -euo pipefail
mod=$1
out=experiments/runs/physiology_semantic_tokenizer/aligned_tokenizer_optimization/20261126_v1
exec >>"$out/$mod.log" 2>&1
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
case "$mod" in
 eeg) pci=00000000:4B:00.0 ;;
 hb) pci=00000000:CA:00.0 ;;
 *) exit 2 ;;
esac
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v b="$pci" '$1==b {print $2}')
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]]
export CUDA_VISIBLE_DEVICES
for action in synthetic baseline full decoder; do
 echo "$(date -Is) $action $mod START"
 .venv/bin/python experiments/scripts/run_aligned_tokenizer_optimization.py "$action" "$mod"
 echo "$(date -Is) $action $mod DONE"
done
