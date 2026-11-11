#!/bin/bash
# Run the paper's method (SteepGS: 3DGS with steepest density control) on all
# benchmark scenes with the official code base in method/SteepGS.
# Output contract: outputs/steepgs/<scene>/point_cloud.ply
# Usage: bash scripts/method.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python data/prepare_scenes.py
python scripts/train_queue.py --runs steepgs "$@"
