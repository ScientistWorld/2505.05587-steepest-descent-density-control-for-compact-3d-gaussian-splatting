#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m redteam.harness shortcut_baseline_inflate
