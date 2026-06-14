# Paper's Method

## Key Contribution

SteepGS introduces Steepest Density Control for 3D Gaussian Splatting. It replaces heuristic densification decisions with per-Gaussian splitting matrices that identify whether splitting can reduce the loss and which offspring displacement gives the steepest local decrease.

## Approach

During training, SteepGS accumulates gradients and a splitting matrix for each Gaussian. At densification intervals, it computes the smallest eigenvalue and eigenvector of each matrix. A Gaussian is split only when the selection conditions hold and the smallest eigenvalue is below a negative threshold. The parent is replaced by two offspring displaced in the positive and negative least-eigenvector directions, with opacity halved for each offspring.

## Main Claims

The primary claim is that this density-control rule reduces Gaussian counts by about 50% without sacrificing rendering quality. Secondary claims are that the rule selects saddle-region primitives, avoids redundant densification, improves memory and rendering efficiency, and can be integrated into the 3DGS CUDA rasterization pipeline.
