#!/bin/bash
set -eu
bus=$1
shift
case "$bus" in 4B|CA) ;; *) exit 2;; esac
id=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F ', ' -v want="00000000:$bus:00.0" '$1==want {print $2}')
test -n "$id"
export CUDA_VISIBLE_DEVICES="$id" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u "$@"
