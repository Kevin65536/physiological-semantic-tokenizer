#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1/cache
/SSD_2/pid-mcm-implementation/.venv/bin/python /SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1/source_snapshot/experiments/scripts/evaluate_ssm_state_continuity.py --run-dir /SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1 --stage pilot
/SSD_2/pid-mcm-implementation/.venv/bin/python /SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1/source_snapshot/experiments/scripts/evaluate_ssm_state_continuity.py --run-dir /SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1 --stage run
