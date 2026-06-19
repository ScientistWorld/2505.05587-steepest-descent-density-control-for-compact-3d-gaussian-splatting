# Evaluation

## Metrics

- PSNR on held-out samples, higher is better.
- Global SSIM proxy on held-out samples, higher is better.
- Mean absolute color error, lower is better.
- Ninety-fifth percentile per-sample L1 error, lower is better.
- Bright-region L1 error on the brightest target quartile, lower is better.
- Mean absolute channel bias, lower is better.
- Final number of Gaussian primitives, lower is better when quality remains competitive. In `scores.json`, this is normalized to paper-equivalent million-point units by mapping the local ADC baseline to the paper's 3.339M-point Mip-NeRF360 ADC reference; raw local counts remain in each run's `outputs/*/metrics.json`.
- Point reduction relative to the adaptive-density baseline, higher is better.
- A hardware-independent rendering-cost proxy proportional to sample-count times primitive-count, lower is better. This score is normalized with the same baseline-relative scale as the point count.
- Total densification splits, lower is better when quality remains competitive.

## Baselines and Targets

The reference target is a quality-versus-compactness comparison: a reconstruction system should use substantially fewer Gaussian primitives than the adaptive-density baseline while preserving competitive held-out reconstruction fidelity. The local benchmark uses a deterministic procedural 2D splatting scene and reports the same metric families in a lightweight form.

## Evaluation Protocol

Run a method so that it writes predictions and metadata under `outputs/`, then run `scripts/evaluate.sh`. The evaluator reads saved prediction artifacts, reconstructs the deterministic benchmark targets from `data/`, and writes `scoring/scores.json`; it does not import method implementations or rerun training.
