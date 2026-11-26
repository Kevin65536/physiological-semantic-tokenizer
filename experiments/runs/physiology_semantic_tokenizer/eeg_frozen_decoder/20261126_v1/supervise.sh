#!/bin/bash
set -euo pipefail
stage=${1:?pilot or train}
case "$stage" in pilot|train) ;; *) exit 2;; esac
run=experiments/runs/physiology_semantic_tokenizer/eeg_frozen_decoder/20261126_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VQ_MMAP_PARENT_CACHE=4096 PYTHONPATH="$PWD"
py=.venv/bin/python
worker() {
    local bus=$1 arm=$2 uuid
    uuid=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F', ' -v bus="$bus" '$1 == bus {print $2}')
    [[ "$uuid" == GPU-* ]] || return 1
    CUDA_VISIBLE_DEVICES="$uuid" "$py" -u experiments/scripts/run_eeg_frozen_decoder.py "$stage" --arm "$arm"
}
worker 00000000:4B:00.0 continued >>"$run/continued.$stage.log" 2>&1 &
first=$!
worker 00000000:CA:00.0 expanded >>"$run/expanded.$stage.log" 2>&1 &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
if (( status )); then exit 1; fi
if [[ "$stage" == train ]]; then
    CUDA_VISIBLE_DEVICES='' "$py" -u experiments/scripts/run_eeg_frozen_decoder.py summarize >>"$run/summary.log" 2>&1
fi
