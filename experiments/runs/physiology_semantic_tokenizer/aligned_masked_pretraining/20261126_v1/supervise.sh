#!/bin/bash
set -euo pipefail
D=experiments/runs/physiology_semantic_tokenizer/aligned_masked_pretraining/20261126_v1
/bin/bash "$D/fit_lane.sh" 4B M3 M2 > "$D/lane_4B.log" 2>&1 &
first=$!
/bin/bash "$D/fit_lane.sh" CA M4 M0_EEG M0_Hb > "$D/lane_CA.log" 2>&1 &
second=$!
rc=0
wait "$first" || rc=1
wait "$second" || rc=1
if [[ $rc != 0 ]]; then exit "$rc"; fi
/bin/bash "$D/worker.sh" 4B experiments/scripts/score_aligned_masked.py > "$D/evaluation.log" 2>&1
