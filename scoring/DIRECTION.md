# Research Direction

## Research Problem

This benchmark studies compact density control for Gaussian splatting: how to decide which primitives should be added, removed, or transformed during optimization so quality remains high without redundant point growth.

## Core Contribution Scope

Future improvements should stay within density-control methods for Gaussian primitive fields. The central object of study is the training-time rule that determines primitive splitting or growth while preserving reconstruction quality.

## Approach Scope

Scientists may modify point selection criteria, local second-order or first-order statistics, offspring placement, opacity handling, pruning, or related optimization schedules. The method should still produce a Gaussian-splatting representation evaluated through the shared primitive-count and reconstruction metrics.

## Out of Bounds

- Replacing Gaussian splatting with a neural network, voxel grid, mesh, or another representation family.
- Improving the score by simply increasing the primitive budget, training samples, or training iterations under the same experiment key.
- Using post-hoc compression alone without addressing density control during training.
