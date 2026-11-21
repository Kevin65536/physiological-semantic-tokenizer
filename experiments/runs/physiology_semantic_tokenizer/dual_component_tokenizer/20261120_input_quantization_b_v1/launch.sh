#!/bin/bash
set -eu
run=$(cd "$(dirname "$0")" && pwd)
exec .venv/bin/python "$run/source/experiments/scripts/run_dual_component_tokenizer.py" --config "$run/source/experiments/configs/physiology_semantic_tokenizer/dual_component_base_matrix_v1.yaml" --project-root "$PWD" --output "$run" --stage "$1"
