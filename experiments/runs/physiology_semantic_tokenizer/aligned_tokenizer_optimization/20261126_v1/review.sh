#!/usr/bin/env bash
set -euo pipefail
mod=$1
d=experiments/runs/physiology_semantic_tokenizer/aligned_tokenizer_optimization/20261126_v1
exec >>"$d/$mod.review.log" 2>&1
while ! test -s "$d/$mod/decoder/summary.json"; do
 systemctl --user is-active --quiet "aligned-opt-1126-$mod.service" || { echo "Producer stopped at $(date -Is)"; exit 1; }
 sleep 45
done
case "$mod" in eeg) bus=00000000:4B:00.0;; hb) bus=00000000:CA:00.0;; *) exit 2;; esac
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v b="$bus" '$1==b {print $2}')
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]]
export CUDA_VISIBLE_DEVICES PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python experiments/scripts/score_aligned_optimization.py "$mod"
