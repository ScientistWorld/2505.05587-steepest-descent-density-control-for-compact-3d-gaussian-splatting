#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Mirrors scripts/evaluate.sh: reads pre-generated method outputs and writes a
# score JSON, but defaults to scoring the test half of the evaluation samples.

set -e

cd /home/user

mkdir -p scoring

python -m eval.test.evaluate_outputs "$@"
