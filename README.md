# Steepest Descent Density Control for Compact 3D Gaussian Splatting

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2505.05587"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2505.05587-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2505.05587](https://arxiv.org/abs/2505.05587) |
| **Domain** | Computer Vision / Neural Rendering |
| **TL;DR** | The paper makes 3D Gaussian Splatting more compact by splitting only primitives whose local splitting matrix indicates a loss-reducing saddle escape direction. |

## Summary

3DGS often creates redundant Gaussian primitives during adaptive densification. SteepGS analyzes densification as a local optimization problem and derives a splitting matrix for each primitive. The optimal split uses two offspring, halves opacity, and moves them along the least-eigenvector direction when the smallest eigenvalue is negative. The paper reports roughly 50% fewer points while preserving rendering quality on Mip-NeRF360, Tanks & Temples, and Deep Blending.

## Problem

How can a Gaussian-splatting radiance-field reconstruction system maintain high novel-view or pixel reconstruction quality while using substantially fewer Gaussian primitives than a standard adaptive-density training pipeline?

## The paper's method

SteepGS introduces Steepest Density Control for 3D Gaussian Splatting. It replaces heuristic densification decisions with per-Gaussian splitting matrices that identify whether splitting can reduce the loss and which offspring displacement gives the steepest local decrease.

## Key results of the paper

- Mip-NeRF360: SteepGS reports 1.606M points versus 3.339M for 3DGS, with PSNR 28.734 versus 29.037.
- Tanks & Temples: SteepGS reports 0.958M points versus 1.830M, with PSNR 23.684 versus 23.743.
- Deep Blending: SteepGS reports 1.296M points versus 2.818M, with PSNR 29.963 versus 29.690.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | -56.4 |
| `gpt-5.6-sol-xhigh` | -114.2 |
| `gpt-5.6-luna-xhigh` | -150.4 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2505.05587 --mode innovate --agent claude-code
sw evaluate 2505.05587
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

