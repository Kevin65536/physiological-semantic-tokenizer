#!/bin/bash
set -eu -o pipefail
exec >> "/SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/observation_factorization/20261120_relations_v2/process.log" 2>&1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
"/SSD_2/pid-mcm-implementation/.venv/bin/python" "/SSD_2/pid-mcm-implementation/experiments/scripts/run_observation_relations.py" --stage preflight --output "/SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/observation_factorization/20261120_relations_v2"
"/SSD_2/pid-mcm-implementation/.venv/bin/python" "/SSD_2/pid-mcm-implementation/experiments/scripts/run_observation_relations.py" --stage screen --output "/SSD_2/pid-mcm-implementation/experiments/runs/physiology_semantic_tokenizer/observation_factorization/20261120_relations_v2"
