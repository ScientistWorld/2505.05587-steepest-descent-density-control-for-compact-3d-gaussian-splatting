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
