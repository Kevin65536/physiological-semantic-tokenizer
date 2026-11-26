#!/bin/bash
set -euo pipefail
D=experiments/runs/physiology_semantic_tokenizer/aligned_masked_pretraining/20261126_v1
bus=$1
shift
for arm in "$@"; do
  /bin/bash "$D/worker.sh" "$bus" experiments/scripts/train_aligned_masked.py train --arm "$arm"
done
