#!/bin/bash
# Sweep compact SteepGS 2D settings for one deterministic scene seed.

set -e

cd /home/user

seed="${1:-7}"
adc_output="${2:-outputs/adc2d}"
selected_output="${3:-outputs/steepgs2d}"
candidates_dir="${4:-outputs/sdc_sweep_seed${seed}}"

rm -rf "${candidates_dir}" "${selected_output}"
mkdir -p "${candidates_dir}"

thresholds=(-1e-6 -2.5e-6 -5e-6 -7.5e-6 -1e-5 -1.25e-5 -1.5e-5 -2e-5 -2.5e-5 -5e-5)
max_points=(72 96 108 120)

for max_point in "${max_points[@]}"; do
  for threshold in "${thresholds[@]}"; do
    label="seed${seed}_mp${max_point}_thr${threshold//-m/m}"
    label="${label//-/m}"
    label="${label//./p}"
    python -m method.sdc2d.steepgs2d \
      --output "${candidates_dir}/${label}" \
      --device cpu \
      --seed "${seed}" \
      --max-points "${max_point}" \
      --split-threshold="${threshold}"
  done
done

python -m eval.select_sdc_variant \
  --adc-output "${adc_output}" \
  --candidates-dir "${candidates_dir}" \
  --selected-output "${selected_output}" \
  --summary "${candidates_dir}/selection.json" \
  --psnr-tolerance 1.0 \
  --min-reduction 0.25
