#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

rm -rf outputs/adc2d outputs/steepgs2d outputs/sdc_sweep

bash scripts/baseline.sh

for threshold in -0.00020 -0.00015 -0.00010 -0.000075 -0.000050 -0.000025; do
  safe_name=$(printf "%s" "$threshold" | tr '-' 'm' | tr '.' 'p')
  python -m method.sdc2d.steepgs2d \
    --output "outputs/sdc_sweep/split_${safe_name}" \
    --device cpu \
    --split-threshold "$threshold"
done

python -m eval.select_sdc_variant \
  --adc-output outputs/adc2d \
  --candidates-dir outputs/sdc_sweep \
  --selected-output outputs/steepgs2d

bash scripts/evaluate.sh
python validate.py
