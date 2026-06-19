#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

rm -rf outputs/compact_synthetic_splatting_seed7 \
       outputs/compact_synthetic_splatting_seed13 \
       scoring/scores.json

run_setting() {
  local experiment_name="$1"
  local seed="$2"
  local append_flag="$3"
  local root="outputs/${experiment_name}"

  rm -rf "${root}"
  mkdir -p "${root}"

  bash scripts/baseline.sh \
    --output "${root}/adc2d" \
    --seed "${seed}"

  SEED="${seed}" \
  ADC_OUTPUT="${root}/adc2d" \
  SWEEP_DIR="${root}/sdc_sweep" \
  SELECTED_OUTPUT="${root}/steepgs2d" \
    bash scripts/sweep_sdc.sh

  if [ "${append_flag}" = "append" ]; then
    bash scripts/evaluate.sh \
      --adc-output "${root}/adc2d" \
      --steepgs-output "${root}/steepgs2d" \
      --experiment-name "${experiment_name}" \
      --append
  else
    bash scripts/evaluate.sh \
      --adc-output "${root}/adc2d" \
      --steepgs-output "${root}/steepgs2d" \
      --experiment-name "${experiment_name}"
  fi
}

run_setting compact_synthetic_splatting_seed7 7 replace
run_setting compact_synthetic_splatting_seed13 13 append
