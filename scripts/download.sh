#!/bin/bash
# Download every external artifact this environment needs (idempotent).
#
# Run on a node with internet access BEFORE building the container or
# submitting compute jobs. Safe to re-run: every file is hash-checked and only
# fetched when missing or different.
#
# Estimated total download size: ~1.25 GB
#   - tandt_db.zip (3DGS-preprocessed Tanks&Temples + Deep Blending scenes, 0.68 GB)
#   - LPIPS VGG-16 backbone (0.55 GB) + LPIPS v0.1 linear heads (7 KB)
# Estimated disk usage after extraction: ~2.0 GB
#
# Layout produced:
#   data/downloads/scenes/<scene>/{sparse/0, images_full}  training views only (method-visible)
#   eval/test/gt/<scene>/*.jpg                               held-out test views (evaluator only)
#   eval/train/gt/<scene>/*.jpg                              validation views (evaluator only)
#   eval/{test,train}/weights/*.pth                          LPIPS weights (evaluator only)
# The archive itself is staged under tmp/ and deleted after extraction, so no
# method-visible location ever holds held-out images.
# Half-resolution training PNGs are built inside jobs by data/prepare_scenes.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARCHIVE_URL="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
ARCHIVE_SHA="816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1"
STAGE="$ROOT/tmp/download_stage"
SCENES="train truck drjohnson playroom"

need_archive=0
for s in $SCENES; do
  [ -f "data/downloads/scenes/$s/sparse/0/images.bin" ] || need_archive=1
  [ -d "data/downloads/scenes/$s/images_full" ] || need_archive=1
  [ -d "eval/test/gt/$s" ] || need_archive=1
  [ -d "eval/train/gt/$s" ] || need_archive=1
done

if [ "$need_archive" = 1 ]; then
  mkdir -p "$STAGE"
  if [ ! -f "$STAGE/tandt_db.zip" ] || ! echo "$ARCHIVE_SHA  $STAGE/tandt_db.zip" | sha256sum -c --status; then
    echo "[download] fetching $ARCHIVE_URL"
    curl -fL --retry 3 -o "$STAGE/tandt_db.zip.part" "$ARCHIVE_URL"
    mv "$STAGE/tandt_db.zip.part" "$STAGE/tandt_db.zip"
  fi
  echo "$ARCHIVE_SHA  $STAGE/tandt_db.zip" | sha256sum -c -
  python3 -I data/split_scenes.py "$STAGE/tandt_db.zip" "$ROOT"
  python3 -I eval/test/fetch_gt.py "$STAGE/tandt_db.zip"
  python3 -I eval/train/fetch_gt.py "$STAGE/tandt_db.zip"
  rm -f "$STAGE/tandt_db.zip"
else
  echo "[download] scenes and held-out views already present"
fi

# fetch <url> <dest> <sha256> [local copy to reuse]
fetch() {
  local url=$1 out=$2 sha=$3 alt=${4:-}
  if [ -f "$out" ] && echo "$sha  $out" | sha256sum -c --status; then return 0; fi
  mkdir -p "$(dirname "$out")"
  if [ -n "$alt" ] && [ -f "$alt" ] && echo "$sha  $alt" | sha256sum -c --status; then
    cp "$alt" "$out.part"
  else
    echo "[download] fetching $url"
    curl -fL --retry 3 -o "$out.part" "$url"
  fi
  mv "$out.part" "$out"
  echo "$sha  $out" | sha256sum -c -
}

VGG_URL="https://download.pytorch.org/models/vgg16-397923af.pth"
VGG_SHA="397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
LIN_URL="https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth"
LIN_SHA="a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868"
fetch "$VGG_URL" eval/test/weights/vgg16-397923af.pth "$VGG_SHA"
fetch "$LIN_URL" eval/test/weights/lpips_vgg_v0.1.pth "$LIN_SHA"
fetch "$VGG_URL" eval/train/weights/vgg16-397923af.pth "$VGG_SHA" eval/test/weights/vgg16-397923af.pth
fetch "$LIN_URL" eval/train/weights/lpips_vgg_v0.1.pth "$LIN_SHA" eval/test/weights/lpips_vgg_v0.1.pth

echo "[download] done"
