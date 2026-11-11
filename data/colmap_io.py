"""Standard-library reader for COLMAP binary models (cameras.bin / images.bin).

Pure Python (struct only) so it runs anywhere ``scripts/download.sh`` runs,
including hosts without numpy. It reproduces the camera conventions used by
the reference 3D Gaussian Splatting loader: R is the transpose of the
world-to-camera rotation obtained from the COLMAP quaternion, T is the COLMAP
translation, and field of view is computed from the PINHOLE focal lengths of
the full-resolution images.
"""

from __future__ import annotations

import io
import math
import struct

CAMERA_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
CAMERA_MODEL_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE"}


def _read(fid, fmt):
    size = struct.calcsize("<" + fmt)
    data = fid.read(size)
    if len(data) != size:
        raise ValueError("truncated COLMAP file")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(data: bytes) -> dict:
    fid = io.BytesIO(data)
    (num,) = _read(fid, "Q")
    cams = {}
    for _ in range(num):
        cam_id, model_id, width, height = _read(fid, "iiQQ")
        n = CAMERA_MODEL_NUM_PARAMS[model_id]
        params = _read(fid, "d" * n)
        cams[cam_id] = {
            "model": CAMERA_MODEL_NAMES.get(model_id, str(model_id)),
            "width": int(width),
            "height": int(height),
            "params": list(params),
        }
    return cams


def read_images_binary(data: bytes) -> dict:
    fid = io.BytesIO(data)
    (num,) = _read(fid, "Q")
    images = {}
    for _ in range(num):
        image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read(fid, "idddddddi")
        name = b""
        while True:
            ch = fid.read(1)
            if ch in (b"\x00", b""):
                break
            name += ch
        (n2d,) = _read(fid, "Q")
        fid.read(24 * n2d)
        images[image_id] = {
            "name": name.decode("utf-8"),
            "qvec": (qw, qx, qy, qz),
            "tvec": (tx, ty, tz),
            "camera_id": cam_id,
        }
    return images


def qvec2rotmat(q):
    w, x, y, z = q
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ]


def focal2fov(focal: float, pixels: float) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def view_records(cameras_bin: bytes, images_bin: bytes) -> list[dict]:
    """Return per-image camera records sorted by image stem (3DGS order)."""
    cams = read_cameras_binary(cameras_bin)
    imgs = read_images_binary(images_bin)
    records = []
    for img in imgs.values():
        cam = cams[img["camera_id"]]
        if cam["model"] == "SIMPLE_PINHOLE":
            fx = fy = cam["params"][0]
        elif cam["model"] == "PINHOLE":
            fx, fy = cam["params"][0], cam["params"][1]
        else:
            raise ValueError(f"unsupported camera model {cam['model']}")
        rot = qvec2rotmat(img["qvec"])
        R = [[rot[j][i] for j in range(3)] for i in range(3)]  # transpose, as in 3DGS
        records.append(
            {
                "name": img["name"].split("/")[-1],
                "stem": img["name"].split("/")[-1].split(".")[0],
                "R": R,
                "T": list(img["tvec"]),
                "fovx": focal2fov(fx, cam["width"]),
                "fovy": focal2fov(fy, cam["height"]),
                "full_width": cam["width"],
                "full_height": cam["height"],
            }
        )
    records.sort(key=lambda r: r["stem"])
    return records


# Split rule shared by every component (fixed, not configurable):
# views are sorted by name; index % 8 == 0 is the held-out test view set used by
# the paper's protocol, index % 8 == 4 is a held-out validation set, every other
# view is a training view.
HOLD = 8
TEST_RESIDUE = 0
VAL_RESIDUE = 4


def split_of(index: int) -> str:
    r = index % HOLD
    if r == TEST_RESIDUE:
        return "test"
    if r == VAL_RESIDUE:
        return "val"
    return "train"
