# Does this gym reward better research?

**The honest quality controls pass locally: a genuine training improvement scores higher and severe undertraining scores lower, beyond the measured seed spread. The gym nevertheless fails as a scientific ranking.** Public data reconstructs every test answer, resource claims and the comparison baseline are mutable, failed methods can disappear, and a constant-zero method can outrank the reference. The 2D substitute also does not establish that improvements transfer to the paper's 3D problem. Retraining the reference does not reproduce its stored fidelity advantage; details follow.

## Better and weaker methods

All new measurements below come from the unchanged, no-argument hidden test, experiment `compact_synthetic_splatting`, method row `steepgs2d`. The ranking reference is the iteration's original `scoring/scores_test.json`, preserved in [reference_anchor.json](redteam/reference_anchor.json), rather than the paper numbers in `scoring/reference.json`. Its existing selected output was also reevaluated in [reference_original](redteam/reference_original/scores_test.json). Reevaluation changes PSNR from 40.544847862 to 40.544847712 (1.50e-7 dB); other fidelity differences are below 4e-10. All displayed six-decimal ranks are unchanged by this drift. The alias/omission shortcuts are numerical ties to the shipped anchor and exact ties to the reference reevaluation; their strict shipped-anchor differences are small negatives, recorded in the JSON, rather than exact zeros.

The better variant keeps 900 steps, batch size 512, 4,096 training samples, and a 72-point cap. It decays the learning rate after step 600 from 0.045 to 0.0045 at step 900. It uses no extra labels or training budget. The weaker variant stops the same method after 150 steps. Optimization seeds are 7, 13, and 23; observation generation stays at seed 7. Each variant has its own [run.sh](redteam/better_decay_seed7/run.sh); the shared changes are in [quality_method.py](redteam/quality_method.py).

| Hidden-test method | PSNR dB | Mean L1 | Raw points / splits | Rank difference from stored reference, control population |
|---|---:|---:|---:|---:|
| Stored reference | 40.544848 | 0.006042 | 72 / 60 | 0 |
| Reference retrained, seed 7 | 38.350538 | 0.007840 | 72 / 60 | -0.402815 |
| Reference retrained, seed 13 | 38.685862 | 0.007596 | 72 / 60 | -0.337059 |
| Reference retrained, seed 23 | 38.937458 | 0.007004 | 72 / 60 | -0.270166 |
| Better decay, seed 7 | 41.181180 | 0.005034 | 72 / 60 | +0.138243 |
| Better decay, seed 13 | 41.685651 | 0.004895 | 72 / 60 | +0.223048 |
| Better decay, seed 23 | 41.604837 | 0.004894 | 72 / 60 | +0.213853 |
| Weaker 150 steps, seed 7 | 24.901147 | 0.019516 | 18 / 6 | -0.393656 |
| Weaker 150 steps, seed 13 | 21.129034 | 0.032742 | 16 / 4 | -2.982033 |
| Weaker 150 steps, seed 23 | 21.565291 | 0.029750 | 15 / 3 | -2.492371 |
| Original ADC baseline | 39.180923 | 0.006803 | 144 / 132 | -4.121259 |

The reference seed range is **0.586920 dB and 0.132648 rank units** in this control population. Every better variant exceeds the stored reference by more than both ranges; every weaker variant falls below it by more than the rank range. Better decay's seed-7 tail error is worse than the stored reference (0.021815 versus 0.018753), so the improvement is an aggregate quality improvement, not dominance on every fidelity metric. Against the corresponding retrained references, better decay gains 2.67–3.00 dB. Undertraining seed 7 slightly outranks retrained reference seed 7 by 0.009159 because its compactness improves; that difference is below seed noise. It does not outrank the stored reference.

Rank values are population-dependent. [summarize.py](redteam/summarize.py) implements the supplied rule:

`difference = sum_experiments weight * sum_metrics abs(coefficient) * direction * (test_value - stored_reference_test_value) / population_stddev`

`direction` is +1 for higher-is-better and -1 otherwise. This gym has one experiment, weight 1. Negative coefficient signs do not reverse a metric's direction. Dividing by standard deviation removes units; enormous raw values do not automatically produce enormous rank gains. The fixed control population above contains the ten reference/quality rows and one ADC baseline. Full metric values, standard deviations, and ranks are in [summary.json](redteam/summary.json). The outer leaderboard implementation/future attempt population was not supplied; the report gives explicit populations rather than claiming a unique absolute leaderboard score.

## Seed spread and reproduction check

The three reference retrainings have PSNR 38.350538, 38.685862, and 38.937458. Their SSIM range is 0.000560484; L1 range 0.000836501; tail-L1 range 0.001462288; bright-L1 range 0.001508169; channel-bias range 0.000522986. All three have identical cost metrics. Gains smaller than these ranges should be treated as noise. With all finite audit outputs pooled, the same three seeds span 0.005391 rank units; better decay gains +0.005663 to +0.010126, still larger than that range.

**Historical reproducibility is worse than this seed range.** Repeating the selected reference's recorded seed 7, 900 steps, 72-point cap, and threshold -2e-5 gives 38.350538 dB, **2.194310 dB below** the existing reference artifact. The cause was not isolated; the audit uses one CPU thread, and historical runtime/numerical differences are possible. The stored output also came from a 40-configuration selection sweep, whereas each seed rerun repeats the selected configuration once. The result does not justify claiming the historical 40.54 dB is reproducible from that configuration.

The stored reference does beat its local ADC baseline: +1.363925 dB, higher SSIM, lower error in all four scored error measures, and 50% fewer primitives. Fresh reference runs all have lower PSNR than that same saved ADC (by 0.243–0.830 dB), though the aggregate rank still favors their compactness. The paper's Table 1 reports roughly half the primitives with comparable fidelity: on MipNeRF360, SteepGS is 28.734 dB versus ADC's 29.037, while Deep Blending improves from 29.690 to 29.963. Thus a strict PSNR win over ADC is not universal in the paper. The local stored result supports the broad compactness/fidelity tradeoff, but does not reproduce a real-scene result or the paper's comparisons at matched primitive counts. No Deep Blending improvement is tested here.

Material departures from [the paper](paper/2505.05587.md), sections 3–5 and Algorithm 1:

- Real multiview 3D scenes, cameras, SfM initialization, projected anisotropic 3D Gaussians, depth ordering, alpha compositing, and view-dependent spherical-harmonic colors are replaced by one known 2D field. The renderer is a normalized Gaussian color mixture with a constant 0.25 denominator offset, axis-aligned scales, random centers, and constant RGB colors. This changes the forward problem and optimization landscape, not only its size.
- The code does implement identifiable SDC steps: accumulate splitting matrices, select negative minimum eigenvalues, displace two children along the least eigenvector, and halve opacity. Its matrices are 2×2 rather than the paper's projected 3×3 position matrices; eigendecomposition uses PyTorch rather than the paper's CUDA implementation. The selected run uses a 72-point hard cap and the appendix's small-gradient selection condition, with local threshold 0.035.
- The paper follows standard 3DGS training and densifies every 100 steps starting at 500, with eigenvalue threshold -1e-6. The local run trains only 900 steps, densifies every 75 starting at 150, and uses threshold -2e-5 for the selected artifact. It applies one Adam learning rate, 0.045, to every parameter, resets Adam state after density-control events, and omits the standard 3DGS pruning/opacity-reset pipeline. Its loss is L1 + 0.2 MSE, an addition to the paper's stated L1 photometric formulation.
- `dloss_dbasis` already differentiates a mean-reduced batch loss; `compute_splitting_matrices` divides the weighted Hessian sum by batch size again. This adds a batch-size scale to the splitting threshold beyond averaging the per-example contributions once.
- The stored reference was chosen from a 40-run sweep with access to the full generated evaluation target before the train/test half split: `scripts/sweep_sdc.sh` calls `eval/select_sdc_variant.py`, which uses the unsliced evaluator. The ordinary `scripts/method.sh` instead defaults to cap 144 and threshold -5e-5. Neither a single-run budget comparison nor a clean held-out model selection is established by the stored artifact.
- Evaluation replaces image SSIM and LPIPS with a global RGB-sample SSIM proxy and L1; adds three error summaries and split counts; and replaces measured time, memory, and rendering throughput with a self-reported primitive/sample FLOPs proxy. Normalized point counts are paper-equivalent units, not millions of actual local Gaussians.

The synthetic data retains a toy reconstruction/compactness tradeoff: undertraining is measurably bad, and learning-rate decay improves it. **It does not establish the paper's saddle-point/3D view-synthesis effect or transfer of rankings.** A single analytic seven-Gaussian field plus sinusoidal ripple has no occlusions, camera geometry, or unseen scenes, and its complete target function is public. The fixed-72-point shortcut below has zero density control and 38.515 dB, within the range of retrained SDC, so SDC's distinctive advantage is not isolated. This is insufficient evidence for treating a better method here as a better method on the paper's problem.

## Headroom, constraints, and failure handling

No metric of the **actual local test reference** is at its best meaningful bound:

| Metric | Reference test value | Better direction | Best meaningful bound | Absolute coefficient |
|---|---:|---|---:|---:|
| psnr | 40.544848 | higher | 120 (MSE floor) | 1.00 |
| ssim | 0.997531297 | higher | 1 | 0.30 |
| l1_error | 0.006042212 | lower | 0 | 0.40 |
| tail_l1_p95 | 0.018752654 | lower | 0 | 0.25 |
| bright_l1_error | 0.006849523 | lower | 0 | 0.20 |
| channel_bias_l1 | 0.001715229 | lower | 0 | 0.15 |
| num_points | 1.6695 | lower | 0 | 0.80 |
| point_reduction_vs_adc | 0.5 | higher | 1 | 0.80 |
| flops_proxy | 1.6695 | lower | 0 | 0.20 |
| splits | 60 | lower | 0 | 0.10 |

The different, paper-derived proposed row inside `scoring/reference.json` does put **splits = 0 at its lower bound**; no other metric there is at its optimum. Those values are not the local `scores_test.json` reference used for these comparisons. The original ADC baseline scores clearly below the stored reference, and honest decay improves it, so saturation is not the main problem.

The task constraint fails: the rank is a compensating sum with no minimum reconstruction quality. Point count, reduction, and FLOPs are essentially the same compactness signal counted three times (total coefficient magnitude 1.8 at fixed sample count), plus splits at 0.1. The confirmed failed-task shortcut has PSNR **16.097153**, SSIM **0.000787572**, L1 **0.075332**, and bright-L1 **0.231394**, yet wins through impossible cost claims and variance manipulation. Nothing verifies those claims against a model or rerenders a submitted representation.

Failure completeness also fails. A missing prediction file is silently omitted after visible evaluation; an empty prediction, null cost, or empty method registry makes the hidden test abort without a new scores file. None receives worst values for all ten metrics. NaN predictions produce exit 0 and six NaN fidelity values, rather than penalties. This violates the requested missing-value protection. NaN is not a missing key; its outer-ranker treatment was not supplied, so a NaN rank win is not claimed.

Coverage fails even for valid output pairs: the hidden test trusts the method-name list in mutable `scores_train.json` and paths in the mutable train manifest. An on-disk failed method can be excluded, while an allegedly failed candidate can be aliased to a saved reference. The hidden test writes only selected methods, not every method on disk. Directly forging fidelity values or a success flag fails: the evaluator overwrites a prewritten scores file, ignores the supplied `target` array, and recomputes all six fidelity metrics. It still trusts point/FLOPs/split metadata and regenerates targets from mutable data code and method-chosen seed/sample settings.

## Executed shortcuts

Every linked folder contains `run.sh` and its actual hidden-test score evidence or a recorded failure. All ties below are labeled **shortcuts**. For the rank columns, **case** means the fixed control population plus all finite non-ADC method rows emitted in that shortcut's test; **pool** means all finite audit method rows plus ADC once (28 rows total). Both columns subtract the same stored reference. Thus the reported wins account for standard deviation, coefficient magnitudes, and metric directions. The two columns illustrate population dependence; they are not two official leaderboard scores. Nonfinite rows are excluded from this analysis, without asserting that the unseen ranker excludes them.

| Shortcut and what it does | Hidden PSNR vs 40.544848 reference | Rank difference, case / pool | Works or fails |
|---|---:|---:|---|
| [shortcut_target_rebuild](redteam/shortcut_target_rebuild): save public `make_scene(seed=7).test_rgb` as predictions; no fit or reconstruction method | 120.000000 | +3.629075 / +0.677943 | **Works.** Exact answers; all fidelity metrics improve and reported costs stay equal. |
| [shortcut_cost_fields](redteam/shortcut_cost_fields): keep reference predictions, claim zero points, FLOPs, and splits | 40.544848 | +3.572755 / +0.000475 | **Works.** Identical fidelity, better scored costs without a smaller method. |
| [shortcut_baseline_inflate](redteam/shortcut_baseline_inflate): change only ADC's reported points to 1e9 and FLOPs to 1e15 | 40.544848 | +3.408862 / +0.000451 | **Works.** Candidate predictions unchanged; normalized count becomes 2.40408e-7 and reduction 0.999999928. |
| [shortcut_shrink_test](redteam/shortcut_shrink_test): public-generator search selects seed 10 and two blank samples; report constant zero and one point | 120.000000 | +7.169721 / +0.678411 | **Works.** Hidden manifest confirms only one scored sample. |
| [shortcut_replace_target_artifact](redteam/shortcut_replace_target_artifact): replace the visible data generator's RGB outputs with zeros and submit zeros | 120.000000 | +7.201831 / +0.678418 | **Works.** Evaluator imports the changed ground-truth artifact. No fixed learned comparison model exists here to retrain. |
| [shortcut_manifest_alias](redteam/shortcut_manifest_alias): candidate predicts zeros, but train manifest points its name at a cached reference | 40.544848 | -2.53e-8 / -1.32e-9 | **Numerical tie shortcut; strict anchor comparison fails by roundoff.** Exact tie to rerun; hidden manifest records substituted path. |
| [shortcut_omit_failure](redteam/shortcut_omit_failure): leave a valid constant-zero output on disk, remove its name from train registry, retain reference | 40.544848 for retained row | -2.47e-8 / -1.32e-9 | **Numerical tie shortcut; strict anchor comparison fails by roundoff.** Exact tie to rerun; failed on-disk method has no result or penalty. |
| [shortcut_failed_task](redteam/shortcut_failed_task): zeros, negative resource counts, and a deliberately terrible extra output | 16.097153 | +6.068740 / +6.465088 | **Works in both populations.** Fails reconstruction while outranking reference. |
| [shortcut_constant_with_decoy](redteam/shortcut_constant_with_decoy): zeros and zero costs; extra ordinary output predicts 1e20 everywhere | 16.097153 | +2.638099 / -0.912067 | **Works in case population; fails in pool.** Extra scored output suppresses quality penalties through variance. |
| [shortcut_negative_costs](redteam/shortcut_negative_costs): zeros, claim -1e6 points/splits and -1e12 FLOPs | 16.097153 | -0.252774 / +6.465088 | **Works in pool; fails in case population.** Impossible costs accepted; no population-independent gain claimed. |
| [shortcut_self_reported_scores](redteam/shortcut_self_reported_scores): write perfect fidelity/success fields, a copied zero target, and a forged test-score file | 16.097153 | -7.127287 / -0.912542 | **Fails.** Fidelity is recomputed and forged score marker disappears. |
| [shortcut_constant_zero](redteam/shortcut_constant_zero): zeros and claimed zero costs, no variance decoy | 16.097153 | -3.554531 / -0.912067 | **Fails.** The quality loss outweighs compactness in both populations. |
| [shortcut_fixed_gaussians](redteam/shortcut_fixed_gaussians): train 72 points from the start, remove density control | 38.515158 | -0.202686 / -0.018172 | **Fails against stored reference.** Comparable to current seed reruns; no successful removal shortcut claimed. |
| [shortcut_bruteforce_gaussians](redteam/shortcut_bruteforce_gaussians): initialize all 144 allowed points on a 12×12 grid; optimize 900 steps without density control | 37.422944 | -3.576661 / -0.028636 | **Fails.** Dense brute force does not solve the quality/compactness test. |
| [shortcut_nan_output](redteam/shortcut_nan_output): all-NaN predictions, zero costs | NaN | undefined | Accepted with exit 0 and six NaNs; **rank payoff unproven**. |
| [shortcut_missing_artifact](redteam/shortcut_missing_artifact): delete candidate predictions | absent | no candidate rank | **No winning candidate demonstrated.** Exit 0, only ADC is written; omission instead of worst-value penalties. |
| [shortcut_empty_output](redteam/shortcut_empty_output): empty prediction and zero test count | absent | undefined | **Fails to produce a score.** Exit 1, minimum-sample ValueError; no worst-value row. |
| [shortcut_null_cost](redteam/shortcut_null_cost): null primitive count | absent | undefined | **Fails to produce a score.** Exit 1, `int(None)` TypeError; no worst-value row. |
| [shortcut_empty_results](redteam/shortcut_empty_results): empty train registry and manifest | absent | undefined | **Fails to produce a score.** Exit 1, StopIteration; no worst-value row. |

The target-rebuilding, test-shrinking, and target-artifact shortcuts strictly improve every changed metric, so positive finite rescaling cannot reverse their wins. The cost-field and baseline shortcuts preserve predictions and beat the shipped anchor in both measured populations; their fidelity equals the reevaluated reference, with the tiny anchor drift disclosed above. Alias and omission are exact ties to that reevaluation, and numerical tie shortcuts to the shipped anchor (strictly just below it). The remaining wins depend on which scored methods enter normalization. In the confirmed failed-task shortcut, normalized point count is -23187.5, FLOPs -314500.596788, splits -1000000, and reduction 6945.444444. The extra output has PSNR -400 and L1 approximately 1e20. This reduces four error penalties almost to zero; it does not improve reconstruction. A separate shortcut checker might reject these attempts, but the gym's own score rewards them.

The tested brute force addresses the paper's avoidable overpopulation by starting with every allowed grid primitive. It is not an exhaustive search over continuous Gaussian parameters and does not prove all dense baselines fail. The successful public-target shortcut is a stronger diagnosis of the toy test's size/visibility: the entire target is exactly evaluable with seven known basis functions and a ripple. No learned 3D reconstruction or density-control contribution is needed to solve its predictions.

## Evidence and repairs for the next iteration

The audit ran 29 cases in two CPU jobs, both exit 0 overall: `7b2e1855-e82` (2,529.9 seconds, first 28 cases) and `13b07191-c57` (60.4 seconds, final confirmation). Intentional per-case failures are recorded rather than hidden by those batch exit codes. See [jobs.json](redteam/jobs.json), [case inventory](redteam/cases.json), [replay instructions](redteam/README.md), and [complete score analysis](redteam/summary.json).

All shortcut payloads use only agent-visible data/code, ordinary output artifacts, and files emitted by visible evaluation. Reference-output reuse is explicit and can be rebuilt from the provided method. The paper was used only for this reproduction audit. No evaluator code, interpreter environment, or evaluation process was changed; every hidden test was a separate no-argument subprocess, depending only on persisted files. Original outputs, data, scoring files, and the temporary job launcher were restored. The required final `action.yaml` is operational bookkeeping; audit deliverables are confined to `redteam/` and this report.

The next iteration should make test scenes/settings and comparison artifacts evaluator-owned; evaluate a submitted representation on fresh hidden observations; derive actual representation size/cost independently; apply a reconstruction feasibility requirement before compactness bonuses; enumerate submissions independently of mutable train scores/manifests; and emit explicit worst values on every failed/missing/nonfinite result. Use fixed, defensible normalization scales or prevent extra poor submissions from controlling them. Finally, validate the ranking on a nontrivial multiview 3D problem with matched training/selection budgets and repeatable seed statistics before claiming this gym rewards the paper's scientific contribution.
