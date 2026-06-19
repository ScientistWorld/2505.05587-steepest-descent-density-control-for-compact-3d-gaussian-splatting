# Verification Report

## Initial Assessment

The workspace claimed `core_claim` and had a validated reproduced result for `compact_synthetic_splatting`: the compact method preserved/improved reconstruction quality while using half as many Gaussian primitives as the local ADC baseline. The milestone is justified for the scaled synthetic 2D benchmark.

## Issues Found

- Minor: `scoring/reference.json`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, and `briefing/evaluation.md` advertised an additional seed-13 experiment that was not present in `scoring/scores.json`.
- Minor: stale submission state in `scripts/run.sh` and `action.yaml` still referenced the unscored plus job.
- Minor: `briefing/evaluation.md` and `scoring/EXPERIMENTS.md` used paper/method phrasing in method-agnostic documentation.
- Cosmetic: a generated root `__pycache__/` directory was present.

## Fixes Applied

- Removed the unscored seed-13 experiment from the reference and target documents, restored the reproduced experiment weight to `1.0`, and removed unused plus-job scripts.
- Reworded method-agnostic evaluation and experiment documentation to describe the task and metrics without exposing the paper's method.
- Updated `scripts/run.sh` to call the stable reproduction path and removed the generated root cache.
- Added notes in `reference.json` explaining which values come from the paper table and which local metrics are lightweight benchmark anchors.
- Re-ran `python validate.py --compare`; validation passed. The remaining train/test warnings are heuristic false positives caused by deterministic split selection, while the evaluators do read saved `predictions.npz` and `metrics.json` artifacts.

## Final Judgment

The workspace is ready as a core-claim research gym for the scaled benchmark. Improve-mode agents can modify the method and be scored through saved output artifacts; from-scratch agents have method-agnostic problem, evaluation, data, targets, constraints, and experiment documentation. The main limitation is that the benchmark is a small deterministic 2D substitute with high absolute reconstruction scores, so it is useful for the central compactness claim but not a full-fidelity 3DGS reproduction.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic 2D benchmark only
