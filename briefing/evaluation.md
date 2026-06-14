# Evaluation

## Metrics

- PSNR on held-out samples, higher is better.
- Global SSIM proxy on held-out samples, higher is better.
- Mean absolute color error, lower is better.
- Final number of Gaussian primitives, lower is better when quality remains competitive.
- Point reduction relative to the adaptive-density baseline, higher is better.
- A hardware-independent rendering-cost proxy proportional to sample-count times primitive-count, lower is better.

## Baselines and Targets

The reference target is the paper's Table 1 claim on Mip-NeRF360: the proposed system reports roughly half as many points as the original 3DGS baseline while preserving similar PSNR/SSIM/LPIPS. The local benchmark uses a deterministic 2D splatting task with an ADC-style baseline and the same output metrics.

## Evaluation Protocol

Run `scripts/reproduce.sh` to train the baseline and proposed method on the same procedural training samples, then run `scripts/evaluate.sh`. The evaluator reads saved predictions and metadata from `outputs/` and writes `scoring/scores.json`; it does not import or rerun the method implementation.
