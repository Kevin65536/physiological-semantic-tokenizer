#!/bin/bash
set -euo pipefail
name=$1
bus=$2
shift 2
root=/home/uais7/pid-mcm-implementation
out=$root/experiments/runs/physiology_semantic_tokenizer/science_ablations/20261126_v1
uuid=""
if [[ $bus != cpu ]]; then
    case $bus in 00000000:4B:00.0|00000000:CA:00.0) ;; *) exit 2;; esac
    uuid=$(nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv,noheader | awk -F ', ' -v card="$bus" '$1==card {print $2}')
    [[ $uuid = GPU-* ]]
fi
systemd-run --user --unit="science-ablations-$name" --working-directory="$root" \
  --setenv=CUDA_VISIBLE_DEVICES="$uuid" --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=PYTHONPATH="$root" --property=LimitNOFILE=8192 \
  --property="StandardOutput=append:$out/$name.log" --property=StandardError=inherit \
  "$root/.venv/bin/python" -u "$@"
systemctl --user show "science-ablations-$name" -p ExecStart -p Environment -p MainPID -p ActiveState > "$out/${name}_unit.txt"
