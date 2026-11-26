#!/bin/bash
set -euo pipefail
run=experiments/runs/physiology_semantic_tokenizer/single_vq_aligned/20261126_v1
entry=experiments/scripts/retrain_single_vq_aligned.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VQ_MMAP_PARENT_CACHE=4096
worker() {
    local mod=$1 bus=$2 uuid
    uuid=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v bus="$bus" '$1==bus {print $2}')
    [[ "$uuid" == GPU-* ]] || return 1
    export CUDA_VISIBLE_DEVICES="$uuid"
    .venv/bin/python -u "$entry" pilot --modality "$mod" || return 1
    .venv/bin/python -u "$entry" train --modality "$mod"
}
worker eeg 00000000:4B:00.0 >>"$run/eeg.log" 2>&1 &
eeg_pid=$!
worker hb 00000000:CA:00.0 >>"$run/hb.log" 2>&1 &
hb_pid=$!
status=0
wait "$eeg_pid" || status=1
wait "$hb_pid" || status=1
if (( status )); then exit 1; fi
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$entry" summarize
