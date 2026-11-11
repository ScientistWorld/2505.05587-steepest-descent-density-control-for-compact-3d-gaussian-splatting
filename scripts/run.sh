#!/bin/bash
# Job script (runs inside the container, workspace at /home/user, GPU, no
# internet). Reproduces the reference end to end: half-resolution training
# images, SteepGS on the four scenes, the ADC and ADC+Thres. baselines, and the
# validation / held-out test evaluation (scoring/scores_{train,test}.json).
# The evaluator constants in eval/*/protocol.json are frozen and not recomputed.
# The hardening audit (variants, shortcuts) is hardening/run_audit.sh.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd /home/user
bash scripts/reproduce.sh
python validate.py
