#!/bin/bash
# Sweep compact SteepGS 2D settings and select the best saved candidate.

set -e

cd /home/user

rm -rf outputs/sdc_sweep outputs/steepgs2d
mkdir -p outputs/sdc_sweep

thresholds=(-1e-6 -2.5e-6 -5e-6 -7.5e-6 -1e-5 -1.25e-5 -1.5e-5 -2e-5 -2.5e-5 -5e-5)
max_points=(72 96 108 120)

for max_point in "${max_points[@]}"; do
  for threshold in "${thresholds[@]}"; do
    label="mp${max_point}_thr${threshold//-m/m}"
    label="${label//-/m}"
    label="${label//./p}"
    python -m method.sdc2d.steepgs2d \
      --output "outputs/sdc_sweep/${label}" \
      --device cpu \
      --max-points "${max_point}" \
      --split-threshold="${threshold}"
  done
done

python -m eval.select_sdc_variant \
  --adc-output outputs/adc2d \
  --candidates-dir outputs/sdc_sweep \
  --selected-output outputs/steepgs2d \
  --summary outputs/sdc_sweep/selection.json \
  --psnr-tolerance 1.0 \
  --min-reduction 0.25
