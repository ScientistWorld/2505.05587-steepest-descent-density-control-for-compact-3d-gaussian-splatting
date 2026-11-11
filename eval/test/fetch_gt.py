"""Populate eval/test/gt/ with the held-out test views (standard library only).

Called by ``scripts/download.sh``. Every extracted JPG is checked against the
sha256 recorded in ``eval/test/views.json``; the evaluator re-checks the same
hashes before scoring, so a modified ground-truth file is detected.

Usage: python3 -I eval/test/fetch_gt.py <tandt_db.zip>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main(zip_path: str) -> None:
    with open(os.path.join(HERE, "views.json")) as f:
        manifest = json.load(f)
    z = zipfile.ZipFile(zip_path)
    for scene, views in manifest["scenes"].items():
        out_dir = os.path.join(HERE, "gt", scene)
        os.makedirs(out_dir, exist_ok=True)
        for v in views:
            out = os.path.join(out_dir, v["name"])
            if os.path.exists(out):
                with open(out, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() == v["sha256"]:
                        continue
            data = z.read(v["archive_path"])
            if hashlib.sha256(data).hexdigest() != v["sha256"]:
                raise SystemExit(f"hash mismatch for {v['archive_path']}")
            with open(out + ".part", "wb") as f:
                f.write(data)
            os.replace(out + ".part", out)
        print(f"[eval/test] {scene}: {len(views)} held-out test views")


if __name__ == "__main__":
    main(sys.argv[1])
