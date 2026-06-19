# Verification Report

## Initial Assessment

The workspace started from a claimed `core_claim` milestone. The underlying reproduced result is valid for the scaled local benchmark: on `compact_synthetic_splatting`, the compact Gaussian-splatting method preserves/improves reconstruction quality while using half as many primitives as the local adaptive-density baseline. However, a pending two-seed expansion had been partially written into the public scoring contract without corresponding completed scores.

## Issues Found

- Critical: `scoring/reference.json`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, `scripts/reproduce.sh`, and the default evaluator experiment name referred to unscored seed-specific experiments, while `scoring/scores.json`, `scores_train.json`, and `scores_test.json` contained only the validated `compact_synthetic_splatting` experiment.
- Critical: the evaluator originally trusted target arrays stored inside method output artifacts. A future method could have fabricated targets inside `predictions.npz` and received inflated scores.
- Minor: `action.yaml` still requested a pending two-seed submit job rather than finalizing the verified `core_claim` state.
- Minor: the login Python lacks NumPy/Torch, so `scripts/evaluate.sh` cannot be executed directly outside the packaged environment. The required dependencies are correctly declared in `environment/setup.sh`.
- Cosmetic/system: `README.md` and `manifest.json` are read-only host-mounted device nodes in this runtime, which makes broad git diff commands fail. They were marked skip-worktree locally; their tracked contents remain intact.

## Fixes Applied

- Restored the scoring contract to the single completed `compact_synthetic_splatting` experiment and set its reference weight back to `1.0`.
- Restored `scripts/reproduce.sh` to the stable validated reproduction path: run the ADC baseline, run the SDC sweep, then evaluate the selected artifacts.
- Reset the default top-level evaluator experiment name to `compact_synthetic_splatting`.
- Updated `eval/`, including train/test evaluators, to recompute deterministic benchmark targets from `data/` and use artifacts only for predictions and metadata.
- Clarified `briefing/evaluation.md` and `scoring/CONSTRAINTS.md` so future methods know that evaluator targets come from tracked data, not from method-supplied arrays.
- Re-ran static checks: shell syntax, Python compilation, `python validate.py --compare`, and targeted diff whitespace checks all pass. The remaining validate warnings are false positives from deterministic train/test split selection; the evaluators do read saved artifacts and do not import `method/`.

## Final Judgment

The workspace is ready as a `core_claim` research gym for the scaled benchmark. Improve-mode agents can modify the method and be scored through saved output artifacts; from-scratch agents have method-agnostic problem, evaluation, data, targets, constraints, and experiment documentation. The main limitation is that this is a deterministic 2D procedural substitute rather than the full 3DGS real-scene benchmark, so the confidence is medium rather than high.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic 2D benchmark only
