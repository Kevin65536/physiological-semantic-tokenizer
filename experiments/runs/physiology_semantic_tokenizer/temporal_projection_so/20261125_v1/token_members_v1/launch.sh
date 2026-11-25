#!/usr/bin/env bash
set -eu
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
gpu_uuid=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F ', ' '$1=="00000000:4B:00.0"{print $2}')
test -n "$gpu_uuid"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
exec .venv/bin/python -u experiments/scripts/audit_projected_so_members.py --device cuda:0
