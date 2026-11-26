#!/usr/bin/env bash
set -eu
stage=$1
p=experiments/runs/physiology_semantic_tokenizer/four_stream_evidence/20261126_v3
unit=four-stream-evidence-v3-${stage}
systemd-run --user --unit="$unit" --working-directory="$PWD" --setenv=PYTHONPATH="$PWD" --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --property=StandardOutput="append:$PWD/$p/$stage.log" --property=StandardError=inherit "$PWD/.venv/bin/python" -u experiments/scripts/run_four_stream_evidence.py "$stage"
systemctl --user show "$unit" -p Id -p Environment -p ExecStart -p WorkingDirectory -p MainPID > "$p/${stage}_unit.txt"
