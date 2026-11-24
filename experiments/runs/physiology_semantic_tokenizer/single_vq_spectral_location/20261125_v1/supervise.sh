#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' '$1=="00000000:4B:00.0" || $1=="00000000:CA:00.0" {print $2}' | paste -sd,)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VQ_MMAP_PARENT_CACHE=4096
exec .venv/bin/python -u experiments/scripts/run_single_vq_spectral_location.py controller
