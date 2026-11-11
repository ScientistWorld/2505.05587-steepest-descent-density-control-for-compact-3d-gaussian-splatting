#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for case in reference_original reference_seed7 reference_seed13 reference_seed23 better_decay_seed7 better_decay_seed13 better_decay_seed23 weaker_steps_seed7 weaker_steps_seed13 weaker_steps_seed23 shortcut_fixed_gaussians shortcut_bruteforce_gaussians shortcut_target_rebuild shortcut_cost_fields shortcut_baseline_inflate shortcut_shrink_test shortcut_replace_target_artifact shortcut_manifest_alias shortcut_omit_failure shortcut_self_reported_scores shortcut_constant_zero shortcut_constant_with_decoy shortcut_negative_costs shortcut_nan_output shortcut_empty_output shortcut_missing_artifact shortcut_null_cost shortcut_empty_results shortcut_failed_task; do
  echo "Running ${case}"
  if ! bash "redteam/${case}/run.sh"; then
    echo "Case failed: ${case}"
  fi
done
python -m redteam.summarize
