"""Extract the training views of the benchmark scenes from the public archive.

Standard library only (runs inside ``scripts/download.sh``). For every scene it
writes the COLMAP model and the full-resolution JPGs of the *training* views to
``data/downloads/scenes/<scene>/``. Held-out views (test and validation, see
``colmap_io.split_of``) are never written here; the evaluator extracts its own
copies into ``eval/``.

Usage: python3 -I data/split_scenes.py <tandt_db.zip> <workspace_root>
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_io  # noqa: E402

SCENES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes.json")


def load_scenes() -> list[dict]:
    with open(SCENES_FILE) as f:
        return json.load(f)["scenes"]


def main(zip_path: str, root: str) -> None:
    z = zipfile.ZipFile(zip_path)
    for scene in load_scenes():
        src = scene["archive_dir"]
        dst = os.path.join(root, "data", "downloads", "scenes", scene["name"])
        sparse_dst = os.path.join(dst, "sparse", "0")
        img_dst = os.path.join(dst, "images_full")
        os.makedirs(sparse_dst, exist_ok=True)
        os.makedirs(img_dst, exist_ok=True)
        for fname in ("cameras.bin", "images.bin", "points3D.bin"):
            with z.open(f"{src}/sparse/0/{fname}") as fin, open(os.path.join(sparse_dst, fname), "wb") as fout:
                shutil.copyfileobj(fin, fout)
        recs = colmap_io.view_records(
            z.read(f"{src}/sparse/0/cameras.bin"), z.read(f"{src}/sparse/0/images.bin")
        )
        n_train = 0
        for idx, rec in enumerate(recs):
            if colmap_io.split_of(idx) != "train":
                continue
            out = os.path.join(img_dst, rec["name"])
            if not os.path.exists(out):
                with z.open(f"{src}/images/{rec['name']}") as fin, open(out + ".part", "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                os.replace(out + ".part", out)
            n_train += 1
        # Remove any held-out image that an older layout may have left behind.
        keep = {r["name"] for i, r in enumerate(recs) if colmap_io.split_of(i) == "train"}
        for fname in os.listdir(img_dst):
            if fname not in keep:
                os.remove(os.path.join(img_dst, fname))
        print(f"[split] {scene['name']}: {n_train} training views of {len(recs)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
