#!/bin/bash
# Job script (runs inside the container, workspace at /home/user, no internet).
#
# Hardening job: trains the reference (SteepGS), the baselines (ADC, ADC+Thres.)
# and the audit variants on the paper's scenes, freezes the evaluator constants
# from the ADC reference (one-time calibration), scores the reference through
# scripts/evaluate.sh, then scores audit variants and shortcut submissions
# through the unchanged held-out evaluator. Re-running resumes: finished
# trainings are skipped. With the audit steps removed this is
# scripts/reproduce.sh.

set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ROOT=/home/user
cd "$ROOT"
mkdir -p hardening logs

nvidia-smi || true
echo "nproc=$(nproc)"
python -c "import torch, diff_gaussian_rasterization, simple_knn; print('torch', torch.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))" || exit 1
python data/prepare_scenes.py || exit 1

# 0. Smoke test of the training stack + renderer parity on a short run (fail-soft).
if [ ! -f hardening/parity_smoke.json ]; then
  SM=checkpoints/smoke
  python method/SteepGS/train.py -s data/downloads/scenes/train -i images -r 1 -m $SM/run \
    --S_estimator inv_cov --densify_strategy steepest --iterations 2000 --test_iterations -1 \
    --save_iterations 2000 --quiet && \
  mkdir -p $SM/ply/train && cp $SM/run/point_cloud/iteration_2000/point_cloud.ply $SM/ply/train/ && \
  python hardening/check_renderer_parity.py $SM/ply hardening/parity_smoke.json 3 || echo "SMOKE/PARITY FAILED"
fi

# 1. Trainings (reference first, then audit variants), all visible GPUs.
python scripts/train_queue.py --groups reference audit --deadline-hours 3.3

# 2. One-time calibration of the evaluator constants from the ADC reference.
if python -c "import json,sys; p=json.load(open('eval/test/protocol.json')); sys.exit(0 if all(c['adc_points'] for c in p['scenes'].values()) else 1)"; then
  echo "protocol already calibrated"
else
  mkdir -p checkpoints/calib_root && ln -sfn "$ROOT/outputs/adc" checkpoints/calib_root/adc
  python -m eval.train.evaluate_outputs --raw --outputs-root checkpoints/calib_root --out hardening/raw_adc_train.json && \
  python -m eval.test.evaluate_outputs --raw --outputs-root checkpoints/calib_root --out hardening/raw_adc_test.json && \
  python scripts/calibrate_protocol.py hardening/raw_adc_train.json hardening/raw_adc_test.json adc || { echo "CALIBRATION FAILED"; exit 1; }
fi

# 3. Renderer parity on the reference outputs.
python hardening/check_renderer_parity.py outputs/steepgs hardening/parity_reference.json 3 || echo "PARITY FAILED"

# 4. Reference scores (what future attempts are ranked against).
bash scripts/evaluate.sh || echo "REFERENCE EVALUATION FAILED"

# 5. Audit variants (gated, both sides). Raw ADC quality is in hardening/raw_adc_*.json.
python -m eval.test.evaluate_outputs --outputs-root hardening/runs --out hardening/scores_test_variants.json || echo "VARIANT TEST EVAL FAILED"
python -m eval.train.evaluate_outputs --outputs-root hardening/runs --out hardening/scores_train_variants.json || echo "VARIANT TRAIN EVAL FAILED"

# 6. Shortcut submissions through the held-out evaluator.
python hardening/make_shortcuts.py || echo "SHORTCUT BUILD FAILED"
for case in hardening/shortcuts/*/; do
  c=$(basename "$case")
  python -m eval.test.evaluate_outputs --outputs-root "$case" --out "hardening/scores_test_shortcut_$c.json" || echo "SHORTCUT $c EVAL FAILED"
done
# Ground-truth tampering, in a sandbox copy of the evaluator (the real GT is untouched).
SB=hardening/sandbox
rm -rf $SB && mkdir -p $SB/eval/test $SB/outputs
cp eval/test/*.py eval/test/*.json $SB/eval/test/ && ln -s "$ROOT/eval/test/weights" $SB/eval/test/weights
cp -r eval/test/gt $SB/eval/test/gt && ln -s "$ROOT/outputs/steepgs" $SB/outputs/candidate
python - <<'EOF'
import json, numpy as np
from PIL import Image
v = json.load(open("hardening/sandbox/eval/test/views.json"))["scenes"]["train"][0]
p = f"hardening/sandbox/eval/test/gt/train/{v['name']}"
w, h = Image.open(p).size
Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(p, quality=95)
EOF
(cd $SB && python -m eval.test.evaluate_outputs --outputs-root outputs --out "$ROOT/hardening/scores_test_shortcut_gt_tamper.json") || echo "GT TAMPER EVAL FAILED"
rm -rf $SB/eval/test/gt

# 7. Summary.
python hardening/summarize.py || echo "SUMMARY FAILED"
python validate.py || true
echo "run.sh finished"
