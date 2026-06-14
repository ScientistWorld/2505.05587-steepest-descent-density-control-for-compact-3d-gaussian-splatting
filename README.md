# Overview

- **Paper ID:** 2505.05587
- **Title:** Steepest Descent Density Control for Compact 3D Gaussian Splatting
- **Domain:** Computer Vision / Neural Rendering
- **TL;DR:** The paper makes 3D Gaussian Splatting more compact by splitting only primitives whose local splitting matrix indicates a loss-reducing saddle escape direction.

## Short Summary

3DGS often creates redundant Gaussian primitives during adaptive densification. SteepGS analyzes densification as a local optimization problem and derives a splitting matrix for each primitive. The optimal split uses two offspring, halves opacity, and moves them along the least-eigenvector direction when the smallest eigenvalue is negative. The paper reports roughly 50% fewer points while preserving rendering quality on Mip-NeRF360, Tanks & Temples, and Deep Blending.

## Key Results

- Mip-NeRF360: SteepGS reports 1.606M points versus 3.339M for 3DGS, with PSNR 28.734 versus 29.037.
- Tanks & Temples: SteepGS reports 0.958M points versus 1.830M, with PSNR 23.684 versus 23.743.
- Deep Blending: SteepGS reports 1.296M points versus 2.818M, with PSNR 29.963 versus 29.690.

---

# Problem Definition

## Research Question

How can a Gaussian-splatting radiance-field reconstruction system maintain high novel-view or pixel reconstruction quality while using substantially fewer Gaussian primitives than a standard adaptive-density training pipeline?

## Why It Matters

Dense Gaussian point clouds increase model storage, memory use, and rendering cost. A successful compact training procedure makes Gaussian splatting more practical for interactive and resource-constrained rendering.

## Success Criteria

A good solution should reconstruct held-out colors with high PSNR/SSIM and low error while reducing the final number of primitives and the compute implied by rendering those primitives. The quality/compactness tradeoff must be evaluated against a baseline trained under the same data and optimization budget.

---

# Paper's Method

## Key Contribution

SteepGS introduces Steepest Density Control for 3D Gaussian Splatting. It replaces heuristic densification decisions with per-Gaussian splitting matrices that identify whether splitting can reduce the loss and which offspring displacement gives the steepest local decrease.

## Approach

During training, SteepGS accumulates gradients and a splitting matrix for each Gaussian. At densification intervals, it computes the smallest eigenvalue and eigenvector of each matrix. A Gaussian is split only when the selection conditions hold and the smallest eigenvalue is below a negative threshold. The parent is replaced by two offspring displaced in the positive and negative least-eigenvector directions, with opacity halved for each offspring.

## Main Claims

The primary claim is that this density-control rule reduces Gaussian counts by about 50% without sacrificing rendering quality. Secondary claims are that the rule selects saddle-region primitives, avoids redundant densification, improves memory and rendering efficiency, and can be integrated into the 3DGS CUDA rasterization pipeline.

---

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

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: done
- **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 0m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 0m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 36m | **GPU**: 0.0h


---

# Reproduction Milestones

**Current: none**

## Stop Justification
- Completed at milestone `none`.
