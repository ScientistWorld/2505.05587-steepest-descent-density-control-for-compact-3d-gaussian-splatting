"""Evaluator-owned reader and renderer for 3D Gaussian splatting PLY files.

The renderer is a pure-PyTorch re-implementation of the forward pass of the
standard tile-based Gaussian rasterizer (Kerbl et al., 2023) so that the
evaluator can re-render every submitted representation itself, on CUDA or CPU:

* EWA projection of each 3D covariance with the Jacobian clamp at 1.3x the
  field of view and a 0.3 px^2 screen-space low-pass;
* 3-sigma screen radius, 16x16 tile binning, per-tile front-to-back depth order;
* alpha = min(0.99, opacity * G); contributions with alpha < 1/255 skipped;
  compositing stops before transmittance would drop below 1e-4;
* view-dependent colour from real spherical harmonics (degree 0-3), +0.5 offset,
  clamped at zero; black background.

The PLY layout is the one written by the reference 3DGS code: x, y, z,
f_dc_0..2, f_rest_* (channel-major), opacity (logit), scale_0..2 (log),
rot_0..3 (quaternion w, x, y, z; normalised here). Any other properties are
ignored.
"""

from __future__ import annotations

import math

import numpy as np
import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

BLOCK = 16
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}
_REQUIRED = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


class InvalidSubmission(ValueError):
    """The submitted representation cannot be scored (counts as a failed run)."""


class GaussianCloud:
    def __init__(self, xyz, opacity_logit, log_scale, rot, sh, degree):
        self.xyz = xyz
        self.opacity_logit = opacity_logit
        self.log_scale = log_scale
        self.rot = rot
        self.sh = sh
        self.degree = degree

    @property
    def num_points(self) -> int:
        return int(self.xyz.shape[0])

    def to(self, device):
        return GaussianCloud(self.xyz.to(device), self.opacity_logit.to(device), self.log_scale.to(device),
                             self.rot.to(device), self.sh.to(device), self.degree)


def read_gaussian_ply(path: str, max_points: int) -> GaussianCloud:
    """Parse a binary little-endian 3DGS PLY. Raises InvalidSubmission."""
    try:
        with open(path, "rb") as f:
            lines = []
            while True:
                raw = f.readline()
                if not raw:
                    raise InvalidSubmission("truncated PLY header")
                if len(lines) > 4096:
                    raise InvalidSubmission("PLY header too long")
                lines.append(raw.decode("ascii").strip())
                if lines[-1] == "end_header":
                    break
            offset = f.tell()
    except (OSError, UnicodeDecodeError) as exc:
        raise InvalidSubmission(f"unreadable PLY: {exc}") from exc
    if not lines or lines[0] != "ply":
        raise InvalidSubmission("not a PLY file")
    if "format binary_little_endian 1.0" not in lines:
        raise InvalidSubmission("PLY must be binary_little_endian 1.0")
    count = None
    props = []
    in_vertex = False
    for line in lines[1:]:
        tok = line.split()
        if not tok or tok[0] in ("comment", "obj_info", "format", "end_header"):
            continue
        if tok[0] == "element":
            if count is None:
                if tok[1] != "vertex" or len(tok) != 3:
                    raise InvalidSubmission("first PLY element must be 'vertex'")
                try:
                    count = int(tok[2])
                except ValueError as exc:
                    raise InvalidSubmission("bad vertex count") from exc
                in_vertex = True
            else:
                in_vertex = False
            continue
        if tok[0] == "property" and in_vertex:
            if len(tok) != 3 or tok[1] not in _PLY_TYPES:
                raise InvalidSubmission(f"unsupported vertex property: {line}")
            props.append((tok[2], _PLY_TYPES[tok[1]]))
    if count is None or count <= 0:
        raise InvalidSubmission("PLY has no Gaussians")
    if count > max_points:
        raise InvalidSubmission(f"PLY has {count} Gaussians, above the hard limit {max_points}")
    names = [p[0] for p in props]
    if len(set(names)) != len(names):
        raise InvalidSubmission("duplicate PLY property names")
    missing = [r for r in _REQUIRED if r not in names]
    if missing:
        raise InvalidSubmission(f"PLY is missing properties {missing}")
    n_rest = 0
    while f"f_rest_{n_rest}" in names:
        n_rest += 1
    if n_rest not in (0, 9, 24, 45):
        raise InvalidSubmission(f"unsupported number of SH rest coefficients: {n_rest}")
    degree = {0: 0, 9: 1, 24: 2, 45: 3}[n_rest]
    dtype = np.dtype(props)
    arr = np.fromfile(path, dtype=dtype, count=count, offset=offset)
    if arr.shape[0] != count:
        raise InvalidSubmission("PLY body shorter than its header declares")

    def col(name):
        return arr[name].astype(np.float64)

    xyz = np.stack([col("x"), col("y"), col("z")], 1)
    opacity = col("opacity")
    scale = np.stack([col(f"scale_{i}") for i in range(3)], 1)
    rot = np.stack([col(f"rot_{i}") for i in range(4)], 1)
    dc = np.stack([col(f"f_dc_{i}") for i in range(3)], 1)[:, None, :]
    if n_rest:
        rest = np.stack([col(f"f_rest_{i}") for i in range(n_rest)], 1)
        rest = rest.reshape(count, 3, n_rest // 3).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest], 1)
    else:
        sh = dc
    with np.errstate(over="ignore", invalid="ignore"):
        parts = [a.astype(np.float32) for a in (xyz, opacity, scale, rot, sh)]
    for a in parts:
        if not np.all(np.isfinite(a)):
            raise InvalidSubmission("PLY contains non-finite parameters")
    t = [torch.from_numpy(np.ascontiguousarray(a)) for a in parts]
    return GaussianCloud(t[0], t[1], t[2], t[3], t[4], degree)


class Camera:
    """Camera in the reference 3DGS convention (row-vector matrices)."""

    def __init__(self, R, T, fovx, fovy, width, height, device):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = np.asarray(R, dtype=np.float64).T
        Rt[:3, 3] = np.asarray(T, dtype=np.float64)
        Rt[3, 3] = 1.0
        c2w = np.linalg.inv(Rt)
        Rt = np.float32(np.linalg.inv(c2w))
        self.world_view = torch.tensor(Rt).transpose(0, 1).to(device)
        znear, zfar = 0.01, 100.0
        tan_y, tan_x = math.tan(fovy / 2), math.tan(fovx / 2)
        top, right = tan_y * znear, tan_x * znear
        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * znear / (2 * right)
        P[1, 1] = 2.0 * znear / (2 * top)
        P[3, 2] = 1.0
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        proj = P.transpose(0, 1).to(device)
        self.full_proj = (self.world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.campos = self.world_view.inverse()[3, :3]
        self.tanfovx = math.tan(fovx * 0.5)
        self.tanfovy = math.tan(fovy * 0.5)
        self.width = int(width)
        self.height = int(height)


def _sh_to_rgb(sh, degree, dirs):
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    res = SH_C0 * sh[:, 0]
    if degree > 0:
        res = res - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            res = (res + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5]
                   + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
                   + SH_C2[3] * xz * sh[:, 7] + SH_C2[4] * (xx - yy) * sh[:, 8])
            if degree > 2:
                res = (res
                       + SH_C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
                       + SH_C3[1] * xy * z * sh[:, 10]
                       + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
                       + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
                       + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
                       + SH_C3[5] * z * (xx - yy) * sh[:, 14]
                       + SH_C3[6] * x * (xx - 3.0 * yy) * sh[:, 15])
    return torch.clamp_min(res + 0.5, 0.0)


@torch.no_grad()
def render(g: GaussianCloud, cam: Camera, max_pairs: int, work_budget: int = 1 << 23, k_chunk: int = 256):
    """Render an HxWx3 float image in [0, inf) (black background)."""
    dev = g.xyz.device
    W, H = cam.width, cam.height
    gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
    wv, fp = cam.world_view, cam.full_proj

    p_view = g.xyz @ wv[:3, :3] + wv[3, :3]
    keep = p_view[:, 2] > 0.2
    idx = torch.nonzero(keep).squeeze(1)
    out = torch.zeros(gy * gx, BLOCK * BLOCK, 3, device=dev)
    if idx.numel() == 0:
        return _untile(out, gx, gy, W, H)
    means = g.xyz[idx]
    pv = p_view[idx]
    p_hom = means @ fp[:3, :] + fp[3, :]
    p_w = 1.0 / (p_hom[:, 3] + 1e-7)
    px_ndc = p_hom[:, 0] * p_w
    py_ndc = p_hom[:, 1] * p_w

    q = torch.nn.functional.normalize(g.rot[idx], dim=1)
    r, x, y, z = q.unbind(1)
    Rm = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], 1).view(-1, 3, 3)
    s = torch.exp(g.log_scale[idx])
    M = Rm * s[:, None, :]
    sigma = M @ M.transpose(1, 2)

    tz = pv[:, 2]
    limx, limy = 1.3 * cam.tanfovx, 1.3 * cam.tanfovy
    tx = torch.clamp(pv[:, 0] / tz, -limx, limx) * tz
    ty = torch.clamp(pv[:, 1] / tz, -limy, limy) * tz
    fx = W / (2.0 * cam.tanfovx)
    fy = H / (2.0 * cam.tanfovy)
    J = torch.zeros(idx.numel(), 2, 3, device=dev)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -(fx * tx) / (tz * tz)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -(fy * ty) / (tz * tz)
    JW = J @ wv[:3, :3].transpose(0, 1)
    cov = JW @ sigma @ JW.transpose(1, 2)
    a = cov[:, 0, 0] + 0.3
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + 0.3
    det = a * c - b * b
    det_inv = 1.0 / det
    con = torch.stack([c * det_inv, -b * det_inv, a * det_inv], 1)
    mid = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp_min(mid * mid - det, 0.1))
    radius = torch.ceil(3.0 * torch.sqrt(torch.maximum(mid + disc, mid - disc)))
    px = ((px_ndc + 1.0) * W - 1.0) * 0.5
    py = ((py_ndc + 1.0) * H - 1.0) * 0.5
    finite = torch.isfinite(con).all(1) & torch.isfinite(radius) & torch.isfinite(px) & torch.isfinite(py)
    ok = (det != 0) & finite
    big = 4.0 * (W + H + 4 * BLOCK)
    radius = torch.nan_to_num(radius, nan=0.0, posinf=big).clamp(max=big)
    pxc = torch.nan_to_num(px, nan=0.0).clamp(-big, big)
    pyc = torch.nan_to_num(py, nan=0.0).clamp(-big, big)
    rminx = torch.clamp(torch.trunc((pxc - radius) / BLOCK), 0, gx)
    rminy = torch.clamp(torch.trunc((pyc - radius) / BLOCK), 0, gy)
    rmaxx = torch.clamp(torch.trunc((pxc + radius + BLOCK - 1) / BLOCK), 0, gx)
    rmaxy = torch.clamp(torch.trunc((pyc + radius + BLOCK - 1) / BLOCK), 0, gy)
    cnt = ((rmaxx - rminx) * (rmaxy - rminy)).long()
    ok &= cnt != 0
    sel = torch.nonzero(ok).squeeze(1)
    if sel.numel() == 0:
        return _untile(out, gx, gy, W, H)

    dirs = means[sel] - cam.campos
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    color = _sh_to_rgb(g.sh[idx[sel]], g.degree, dirs)
    depth = tz[sel]
    order = torch.argsort(depth, stable=True)
    sel = sel[order]
    color = color[order]
    xy = torch.stack([px[sel], py[sel]], 1)
    con = con[sel]
    opac = torch.sigmoid(g.opacity_logit[idx[sel]])
    rminx, rminy, rmaxx = rminx[sel].long(), rminy[sel].long(), rmaxx[sel].long()
    cnt = cnt[sel]
    total = int(cnt.sum())
    if total > max_pairs:
        raise InvalidSubmission(f"render workload {total} Gaussian-tile pairs exceeds limit {max_pairs}")

    n = sel.numel()
    gid = torch.repeat_interleave(torch.arange(n, device=dev), cnt)
    start = torch.cumsum(cnt, 0) - cnt
    local = torch.arange(total, device=dev) - start[gid]
    wt = (rmaxx - rminx)[gid]
    tile = (rminy[gid] + local // wt) * gx + (rminx[gid] + local % wt)
    del local, wt, start
    tile_sorted, perm = torch.sort(tile, stable=True)
    gid = gid[perm]
    del perm, tile
    tile_cnt = torch.bincount(tile_sorted, minlength=gx * gy)
    tile_start = torch.cumsum(tile_cnt, 0) - tile_cnt
    del tile_sorted

    pix = torch.arange(BLOCK * BLOCK, device=dev)
    off_x = (pix % BLOCK).float()
    off_y = (pix // BLOCK).float()
    nonempty = torch.nonzero(tile_cnt).squeeze(1)
    nonempty = nonempty[torch.argsort(tile_cnt[nonempty], descending=True, stable=True)]
    counts_sorted = tile_cnt[nonempty].tolist()
    i = 0
    T_MIN = 0.0001
    A_MIN = 1.0 / 255.0
    while i < len(counts_sorted):
        kmax = counts_sorted[i]
        kc = min(kmax, k_chunk)
        nb = max(1, work_budget // (BLOCK * BLOCK * kc))
        tiles = nonempty[i:i + nb]
        i += nb
        starts = tile_start[tiles]
        cnts = tile_cnt[tiles]
        pxf = ((tiles % gx) * BLOCK).float()[:, None] + off_x[None, :]
        pyf = ((tiles // gx) * BLOCK).float()[:, None] + off_y[None, :]
        T = torch.ones(tiles.numel(), BLOCK * BLOCK, device=dev)
        C = torch.zeros(tiles.numel(), BLOCK * BLOCK, 3, device=dev)
        done = (pxf >= W) | (pyf >= H)  # pixels outside the image never composite
        act = torch.arange(tiles.numel(), device=dev)
        for k0 in range(0, kmax, kc):
            live = (cnts[act] > k0) & ~done[act].all(1)
            act = act[live]
            if act.numel() == 0:
                break
            ks = torch.arange(k0, min(k0 + kc, kmax), device=dev)
            valid = ks[None, :] < cnts[act][:, None]
            pos = torch.clamp(starts[act][:, None] + ks[None, :], max=total - 1)
            gk = gid[pos]
            dx = xy[gk, 0][:, None, :] - pxf[act][:, :, None]
            dy = xy[gk, 1][:, None, :] - pyf[act][:, :, None]
            ck = con[gk]
            power = -0.5 * (ck[:, None, :, 0] * dx * dx + ck[:, None, :, 2] * dy * dy) - ck[:, None, :, 1] * dx * dy
            del dx, dy
            alpha = torch.clamp_max(opac[gk][:, None, :] * torch.exp(power), 0.99)
            use = (power <= 0) & (alpha >= A_MIN) & valid[:, None, :]
            alpha = torch.where(use, alpha, torch.zeros((), device=dev))
            del power, use
            Ta = T[act]
            cum = Ta[:, :, None] * torch.cumprod(1.0 - alpha, dim=2)
            excl = torch.cat([Ta[:, :, None], cum[:, :, :-1]], dim=2)
            hit = alpha > 0
            contrib = hit & (cum >= T_MIN) & ~done[act][:, :, None]
            wgt = torch.where(contrib, alpha * excl, torch.zeros((), device=dev))
            C[act] += torch.bmm(wgt, color[gk])
            T[act] = Ta - wgt.sum(2)
            done[act] |= (hit & (cum < T_MIN)).any(2)
            del alpha, cum, excl, hit, contrib, wgt
        out[tiles] = C
    return _untile(out, gx, gy, W, H)


def _untile(out, gx, gy, W, H):
    img = out.view(gy, gx, BLOCK, BLOCK, 3).permute(0, 2, 1, 3, 4).reshape(gy * BLOCK, gx * BLOCK, 3)
    return img[:H, :W]
