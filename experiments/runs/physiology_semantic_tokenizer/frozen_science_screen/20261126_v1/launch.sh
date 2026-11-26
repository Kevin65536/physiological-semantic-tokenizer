#!/bin/bash
set -euo pipefail
stage=$1
shift
root=/home/uais7/pid-mcm-implementation
cd "$root"
out=$root/experiments/runs/physiology_semantic_tokenizer/frozen_science_screen/20261126_v1
systemd-run --user --unit="frozen-science-v1-$stage" --working-directory="$root" \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=PYTHONPATH="$root" \
  --property="StandardOutput=append:$out/$stage.log" --property=StandardError=inherit \
  "$root/.venv/bin/python" -u experiments/scripts/run_frozen_science_screen.py "$stage" "$@"
