#!/bin/bash
# Baselines on all benchmark scenes (official 3DGS / SteepGS code base):
#   outputs/adc        standard 3DGS adaptive density control
#   outputs/adc_thres  ADC whose densification stops at SteepGS's Gaussian count
#                      on the same scene ("3DGS + Thres." in the paper); needs
#                      outputs/steepgs, so run scripts/method.sh first.
# Usage: bash scripts/baseline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python data/prepare_scenes.py
python scripts/train_queue.py --runs adc adc_thres "$@"
