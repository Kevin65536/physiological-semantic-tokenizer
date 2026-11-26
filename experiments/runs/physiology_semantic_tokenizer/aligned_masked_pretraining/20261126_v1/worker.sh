#!/bin/bash
set -euo pipefail
bus=$1
shift
case "$bus" in 4B|CA) ;; *) exit 4;; esac
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v id="00000000:${bus}:00.0" '$1==id {print $2}')
[[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]]
export CUDA_VISIBLE_DEVICES OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
exec .venv/bin/python -u "$@"
