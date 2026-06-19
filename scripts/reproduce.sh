#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction.

set -e

cd /home/user

rm -rf outputs/adc2d outputs/steepgs2d outputs/sdc_sweep scoring/scores.json

bash scripts/baseline.sh
bash scripts/sweep_sdc.sh
bash scripts/evaluate.sh
