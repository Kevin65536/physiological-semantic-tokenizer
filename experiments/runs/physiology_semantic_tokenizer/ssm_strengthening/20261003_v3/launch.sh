#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v3/cache
/SSD_2/pid-mcm-implementation/.venv/bin/python /SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v3/source_snapshot/experiments/scripts/evaluate_ssm_strengthening.py --project-root /SSD_2/pid-mcm-implementation --run-dir /SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v3 --source-run /SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v2 --workers 1 --stage correct-ridge
