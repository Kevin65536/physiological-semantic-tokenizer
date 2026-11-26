#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VQ_MMAP_PARENT_CACHE=4096 PYTHONPATH="$PWD"
d=experiments/runs/physiology_semantic_tokenizer/eeg_encoder_diagnostics/20261126_v1
worker() {
    local bus=$1 arch=$2 fit=$3 uuid
    uuid=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v bus="$bus" '$1==bus {print $2}')
    [[ "$uuid" == GPU-* ]] || return 1
    export CUDA_VISIBLE_DEVICES="$uuid"
    while ! test -f "$d/$arch/models/eeg__waveform__warmup/summary.json"; do
        if test -f "$d/$arch/models/eeg__waveform__warmup/failure.json"; then return 1; fi
        sleep 3
    done
    for phase in train score; do
        date -Is
        .venv/bin/python -u experiments/scripts/run_eeg_encoder_diagnostics.py "$phase" --arch "$arch" --fit "$fit" || return 1
    done
}
worker 00000000:4B:00.0 observed hard >>"$d/observed_hard_worker.log" 2>&1 &
a=$!
worker 00000000:CA:00.0 wide continuous >>"$d/wide_continuous_worker.log" 2>&1 &
b=$!
worker 00000000:4B:00.0 deep continuous >>"$d/deep_continuous_worker.log" 2>&1 &
c=$!
status=0
for pid in "$a" "$b" "$c"; do wait "$pid" || status=1; done
exit "$status"
