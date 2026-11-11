#!/bin/bash
set -euo pipefail
cd /home/user
bash redteam/batch.sh 2>&1 | tee redteam/batch.log
