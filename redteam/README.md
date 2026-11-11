# Scoring audit evidence

Read [../REDTEAM.md](../REDTEAM.md) for the findings. This directory contains 29 executed cases: ten reference/quality controls and nineteen shortcut probes, including two unsuccessful algorithm-removal/brute-force probes. Every case has its own `run.sh`.

Run cases only on a compute node, from `/home/user`, in the gym's existing container:

```bash
bash redteam/shortcut_target_rebuild/run.sh
# Or replay the entire audit:
bash redteam/batch.sh
```

Each case installs an ordinary method artifact, runs `bash scripts/evaluate_train.sh`, then runs **`bash scripts/evaluate_test.sh` with no arguments in a new subprocess**. The installer uses visible data, metadata, generated train scores/manifests, and the provided method/baseline. It does not import hidden test modules. It changes no evaluator, interpreter, installed package, import path, or evaluator process. The per-case scripts limit CPU thread counts for the one-CPU jobs.

`harness.py` temporarily moves the original `outputs/` aside and restores outputs, the data generator, and all four scoring/manifest files in `finally`. Cost/output-only probes reuse the original trained outputs. If outputs are absent on a fresh clone, the harness first trains the visible ADC and recorded selected reference configuration; the historical reference score is not guaranteed to reproduce (see the report). The audit used the original existing outputs. Improve mode supplies the method needed by this bootstrap. The target-rebuilding and target-setting shortcuts themselves require no paper or hidden labels.

Evidence in each case directory:

- `scores_test.json`: the actual file emitted by the hidden test, when one was emitted. Values live under `experiments.compact_synthetic_splatting.results`.
- `scores_train.json` and both evaluation manifests, when emitted.
- `result.json`: exits, original metadata, installation details, and directories present at test time. Some original metadata contains Python's nonstandard `NaN` value; it is preserved as evidence.
- `audit_rank.json`: analysis of the saved test results. It records a missing candidate, nonfinite metric, or last exception when a finite rank cannot be computed. Failed cases intentionally have no invented `scores_test.json`.

`quality_method.py` makes the honest training changes and the two algorithm controls. `summarize.py` performs lightweight, standard-library analysis of saved test scores only; run `python -m redteam.summarize` to regenerate `summary.json` and `audit_rank.json`. No analysis values are fed to the evaluator. `reference_anchor.json` preserves the iteration's original `scoring/scores_test.json`, which is the ranking target. `ranking_config.json` preserves the metric configuration, including original coefficient signs; the analysis uses their magnitudes and the specified better directions. The report discloses tiny floating-point drift when reevaluating the original artifact.

`jobs.json` records the two completed CPU jobs. Raw training logs, prediction arrays, Python caches, and temporary restoration state are ignored. Small score evidence is tracked. A replay with a regenerated reference can change the numerical results and replaces the saved evidence for that case.
