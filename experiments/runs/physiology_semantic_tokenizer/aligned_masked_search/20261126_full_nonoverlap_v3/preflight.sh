#!/bin/bash
set -euo pipefail
export ALIGNED_SEARCH_PANEL=full
D=experiments/runs/physiology_semantic_tokenizer/aligned_masked_search/20261126_full_nonoverlap_v3
P=experiments/scripts/run_aligned_masked_search.py
worker=experiments/runs/physiology_semantic_tokenizer/aligned_masked_pretraining/20261126_v1/worker.sh
export PYTHONPATH="$PWD"
/bin/bash "$worker" 4B "$P" prepare --split train >>"$D/prepare.log" 2>&1
/bin/bash "$worker" 4B "$P" pilot --lane 0 >>"$D/pilot0.log" 2>&1 &
first=$!
/bin/bash "$worker" CA "$P" pilot --lane 1 >>"$D/pilot1.log" 2>&1 &
second=$!
rc=0
wait "$first" || rc=1
wait "$second" || rc=1
exit "$rc"
