#!/bin/bash
# Sweep compact SteepGS 2D settings and select the best saved candidate.
#
# Environment overrides:
#   SEED=13
#   ADC_OUTPUT=outputs/seed13/adc2d
#   SWEEP_DIR=outputs/seed13/sdc_sweep
#   SELECTED_OUTPUT=outputs/seed13/steepgs2d

set -e

cd /home/user

seed="${SEED:-7}"
adc_output="${ADC_OUTPUT:-outputs/adc2d}"
sweep_dir="${SWEEP_DIR:-outputs/sdc_sweep}"
selected_output="${SELECTED_OUTPUT:-outputs/steepgs2d}"

rm -rf "${sweep_dir}" "${selected_output}"
mkdir -p "${sweep_dir}"

thresholds=(-1e-6 -2.5e-6 -5e-6 -7.5e-6 -1e-5 -1.25e-5 -1.5e-5 -2e-5 -2.5e-5 -5e-5)
max_points=(72 96 108 120)

for max_point in "${max_points[@]}"; do
  for threshold in "${thresholds[@]}"; do
    label="mp${max_point}_thr${threshold//-m/m}"
    label="${label//-/m}"
    label="${label//./p}"
    python -m method.sdc2d.steepgs2d \
      --output "${sweep_dir}/${label}" \
      --device cpu \
      --seed "${seed}" \
      --max-points "${max_point}" \
      --split-threshold="${threshold}"
  done
done

python -m eval.select_sdc_variant \
  --adc-output "${adc_output}" \
  --candidates-dir "${sweep_dir}" \
  --selected-output "${selected_output}" \
  --summary "${sweep_dir}/selection.json" \
  --psnr-tolerance 1.0 \
  --min-reduction 0.25
