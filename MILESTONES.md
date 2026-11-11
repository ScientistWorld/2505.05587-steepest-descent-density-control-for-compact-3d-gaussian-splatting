# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-19] - core_claim (2D procedural substitute; superseded)
- Earlier iterations reproduced a compact-splitting effect on a 2D toy field. The redteam audit (REDTEAM.md) showed its test was regenerable from public code and its reference was not reproducible.

### [2026-11-11] - core_claim_plus (hardening iteration, paper's real setup)
- Rebuilt on the paper's Tanks&Temples (train, truck) and Deep Blending (drjohnson, playroom) scenes with the official SteepGS/3DGS code, half resolution, paper's every-8th-view test split.
- Restored the paper's off-spring opacity halving (the released code had it behind an inverted flag).
- Held-out test: SteepGS 26.982 dB / SSIM 0.8925 / LPIPS 0.1448 with 0.465x the Gaussians of standard 3DGS (26.662 / 0.8912 / 0.1404); paper on the same scenes: 0.488x at +0.11 dB. Three seeds span 0.12 dB.
- Secondary claim (Table 1, "3DGS + Thres."): standard densification stopped at the same count loses 2.5 dB on train and fails the quality floor.
- Ablation: removing the paper's opacity halving raises the count to 0.63x and lowers PSNR by 0.22 dB.
- Not reproduced: Mip-NeRF 360, 3DGS-MCMC / Revising-GS comparisons, training time / memory / FPS.

## Stop Justification
- Completed at milestone `core_claim_plus`: the central claim and the matched-budget comparison hold on two of the paper's datasets with seeds; further claims need Mip-NeRF 360 and other code bases.
