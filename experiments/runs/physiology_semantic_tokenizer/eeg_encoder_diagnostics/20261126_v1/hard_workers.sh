#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VQ_MMAP_PARENT_CACHE=4096 PYTHONPATH="$PWD"
d=experiments/runs/physiology_semantic_tokenizer/eeg_encoder_diagnostics/20261126_v1
worker() {
    local bus=$1 arch=$2 uuid
    uuid=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v bus="$bus" '$1==bus {print $2}')
    [[ "$uuid" == GPU-* ]] || return 1
    export CUDA_VISIBLE_DEVICES="$uuid"
    for phase in train score; do
        date -Is
        .venv/bin/python -u experiments/scripts/run_eeg_encoder_diagnostics.py "$phase" --arch "$arch" --fit hard || return 1
    done
}
worker 00000000:4B:00.0 wide >>"$d/wide_hard_worker.log" 2>&1 &
first=$!
worker 00000000:CA:00.0 deep >>"$d/deep_hard_worker.log" 2>&1 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
exit "$status"
