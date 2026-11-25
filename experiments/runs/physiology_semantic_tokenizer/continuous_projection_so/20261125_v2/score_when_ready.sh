#!/bin/bash
set -eu
root=experiments/runs/physiology_semantic_tokenizer/continuous_projection_so/20261125_v2
seed=$1
bus=$2
while [ ! -f "$root/seed_$seed/C/summary.json" ]; do
    systemctl --user is-active --quiet continuous-so-formal-v2 || exit 1
    sleep 10
done
exec bash "$root/gpu.sh" "$bus" experiments/scripts/score_continuous_so_formal.py continuous "$seed"
