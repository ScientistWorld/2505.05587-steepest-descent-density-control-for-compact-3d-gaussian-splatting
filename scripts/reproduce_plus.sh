#!/bin/bash
# Reproduce the core claim on the default scene and one additional scene seed.

set -e

cd /home/user

rm -rf \
  outputs/adc2d \
  outputs/steepgs2d \
  outputs/adc2d_seed13 \
  outputs/steepgs2d_seed13 \
  outputs/sdc_sweep \
  outputs/sdc_sweep_seed13 \
  scoring/scores.json

bash scripts/baseline.sh
bash scripts/sweep_sdc.sh
bash scripts/evaluate.sh --experiment-name compact_synthetic_splatting

python -m baseline.sdc2d.adc2d \
  --output outputs/adc2d_seed13 \
  --device cpu \
  --seed 13

bash scripts/sweep_sdc_seed.sh 13 outputs/adc2d_seed13 outputs/steepgs2d_seed13 outputs/sdc_sweep_seed13

bash scripts/evaluate.sh \
  --adc-output outputs/adc2d_seed13 \
  --steepgs-output outputs/steepgs2d_seed13 \
  --experiment-name compact_synthetic_splatting_seed13 \
  --append
