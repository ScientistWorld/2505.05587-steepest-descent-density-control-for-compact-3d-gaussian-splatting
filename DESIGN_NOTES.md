# Design Notes

Checked the train/test evaluation split for shortcut resistance, method attachment, leakage, script copy-paste errors, and scoring schema completeness.

Changes made:

- Added `tail_l1_p95`, `bright_l1_error`, and `channel_bias_l1` to both `eval/train/evaluate_outputs.py` and `eval/test/evaluate_outputs.py`. These metrics still read only saved method artifacts, but they make it harder to game global PSNR/L1 by fitting easy regions while leaving localized or systematic color errors.
- Added required `higher_is_better`, `coefficient`, and `description` fields for every nonzero metric in `scoring/reference.json`; existing metric entries were missing descriptions.
- Kept the existing method enumeration behavior: train discovers all available output directories or explicit `--method-output` entries, while test can reuse the train manifest when present and otherwise independently discovers outputs. No hardcoded single method default was introduced.

Static checks performed:

- `bash -n scripts/evaluate_train.sh` and `bash -n scripts/evaluate_test.sh` passed.
- Grepped for forbidden paper-agreement metric names and hardcoded method defaults; none were found in the train/test split files.
- Confirmed `scripts/evaluate_test.sh` invokes `eval.test.evaluate_outputs` and defaults to `scoring/scores_test.json`.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.

Runtime concerns for the runner iteration:

- I did not run evaluators or experiments in this design-only pass. The runner should verify that the new distribution-sensitive metrics have non-degenerate ranges on real artifacts and do not saturate near zero or one method for all runs.
- The split currently partitions saved test samples after method inference. This is appropriate for detecting overfit to the visible score subset, but it is still a sample split over the same procedural scene rather than an OOD scene split.
- The test evaluator optionally reads `scoring/scores_train.json` and `scoring/evaluation_manifest_train.json` to align method names and paths. This does not expose test metrics, but the runner should ensure stale train manifests are removed or regenerated before final held-out scoring.
