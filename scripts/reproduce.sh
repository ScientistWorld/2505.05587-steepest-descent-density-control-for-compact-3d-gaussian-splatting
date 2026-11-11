#!/bin/bash
# Reproduce the reference end to end (inside the container, GPU, no internet).
# Assumes scripts/download.sh has run on a node with internet.
#   1. half-resolution training images          (data/prepare_scenes.py)
#   2. SteepGS on all scenes                     (scripts/method.sh)
#   3. ADC and ADC+Thres. baselines              (scripts/baseline.sh)
#   4. validation + held-out test evaluation     (scripts/evaluate.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate.sh
