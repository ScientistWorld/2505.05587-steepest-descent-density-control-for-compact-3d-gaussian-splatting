# Paper's Method

## Key Contribution

SteepGS introduces Steepest Density Control (SDC) for 3D Gaussian Splatting. It
treats densification as escaping saddle points of the photometric loss: for each
Gaussian it accumulates a *splitting matrix* S = E[(dl/dsigma) Hessian of the
Gaussian w.r.t. its own parameters] (position only, 3x3). Splitting can lower the
loss only if S has a negative eigenvalue; the optimal split uses two
off-springs, each with half the parent's opacity, displaced along +/- the
eigenvector of the smallest eigenvalue.

## Implementation in this gym (`method/SteepGS`, official code base)

- `python method/SteepGS/train.py -s <scene> -i images -r 1 --densify_strategy steepest --S_estimator inv_cov`
  (wrapped by `scripts/method.sh` / `scripts/train_queue.py`), 30k steps, standard
  3DGS optimiser and schedule, density control every 100 steps from 500 to 15k,
  eigenvalue threshold -1e-6.
- Selection = ADC's view-space gradient condition AND lambda_min(S) < -1e-6;
  off-springs are placed along the least eigenvector (length drawn from the
  parent's scale for large Gaussians, learning-rate-sized steps for small ones).
- The splitting matrices are accumulated by the CUDA rasteriser's backward pass
  (`inv_cov` estimator, the code base's documented default).
- Opacity halving: the released code applied it only when the option
  `no_div_opacity` was passed (inverted flag) and never to the parent in the
  clone branch. The gym restores the paper's rule (Theorem 2, Algorithm 1): both
  off-springs carry half the parent's opacity; `no_div_opacity` now turns the
  halving off. Measured effect on this benchmark: without halving the method
  keeps 0.63x of the standard Gaussian count at 26.76 dB; with the paper's
  halving 0.47x at 26.98 dB.
- Other changes to the official code are plumbing only: `--seed`/`--gpu`, a
  loader that skips held-out views and reads the half-resolution PNGs,
  `--densify_max_points` (the paper's "3DGS + Thres." baseline), an optional
  `--final_prune_opacity` schedule (off by default), and two CUDA 12 include
  fixes.

## Main Claims

About 50% fewer Gaussians than standard 3DGS without sacrificing rendering
quality (Tables 1-5), and better quality than standard densification stopped at
the same Gaussian count (Table 1, "3DGS + Thres."). Secondary claims (training
time, memory, FPS; comparisons with 3DGS-MCMC, Revising-GS; Mip-NeRF 360) are not
part of this gym.
