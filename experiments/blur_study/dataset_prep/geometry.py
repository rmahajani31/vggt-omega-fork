"""Cross-view depth-reprojection checks shared by the blur-study dataset converters.

Follows training/dataset_preparation/download_and_convert.md ("Geometry validation on real
data"): unproject valid depth of frame i with its intrinsics and the loader's pixel-centre
convention, map it through world into frame j, and compare the projected z with frame j's
depth. Competing interpretations (pose direction, axis flips, range vs z-depth, ...) are
scored on the same pairs so the documented convention must clearly beat the alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def to_4x4(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[: matrix.shape[0], :4] = matrix[:, :4]
    return out


def pixel_grid(height: int, width: int, stride: int = 1):
    """Pixel centres (u + 0.5, v + 0.5), the convention used by the training loader."""
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    return u.astype(np.float64) + 0.5, v.astype(np.float64) + 0.5, v, u


def unproject(depth: np.ndarray, K: np.ndarray, stride: int = 4):
    u, v, rows, cols = pixel_grid(*depth.shape, stride)
    z = depth[rows, cols].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1)[valid]


@dataclass
class PairScore:
    overlap: float        # fraction of source samples landing in-bounds with valid target depth
    inlier: float         # fraction of overlapping samples within rel_tol of target depth
    median_rel: float     # median |z_proj - z_target| / z_target over overlapping samples
    samples: int


def reproject_score(depth_i, K_i, cw_i, depth_j, K_j, cw_j, stride=4, rel_tol=0.01) -> PairScore:
    """Score camera-from-world poses ``cw_*`` (3x4 or 4x4) against two depth maps."""
    points_cam_i = unproject(depth_i, K_i, stride)
    if len(points_cam_i) == 0:
        return PairScore(0.0, 0.0, float("nan"), 0)
    T_i, T_j = to_4x4(cw_i), to_4x4(cw_j)
    homogeneous = np.concatenate([points_cam_i, np.ones((len(points_cam_i), 1))], 1)
    points_cam_j = (T_j @ np.linalg.inv(T_i) @ homogeneous.T).T[:, :3]
    z = points_cam_j[:, 2]
    front = z > 1e-6
    u = K_j[0, 0] * points_cam_j[:, 0] / np.where(front, z, 1) + K_j[0, 2]
    v = K_j[1, 1] * points_cam_j[:, 1] / np.where(front, z, 1) + K_j[1, 2]
    h, w = depth_j.shape
    cols, rows = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inside = front & (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    target = np.zeros_like(z)
    target[inside] = depth_j[rows[inside], cols[inside]]
    overlap = inside & np.isfinite(target) & (target > 0)
    if not overlap.any():
        return PairScore(0.0, 0.0, float("nan"), len(z))
    rel = np.abs(z[overlap] - target[overlap]) / target[overlap]
    return PairScore(float(overlap.mean()), float((rel < rel_tol).mean()), float(np.median(rel)), len(z))


def range_to_z(range_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Euclidean ray distance -> planar z for a pinhole camera (pixel-centre convention)."""
    u, v, _, _ = pixel_grid(*range_map.shape, 1)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return (range_map / np.sqrt(x * x + y * y + 1.0)).astype(np.float32)


def rotation_report(cam_from_worlds: np.ndarray) -> dict:
    R = cam_from_worlds[:, :3, :3].astype(np.float64)
    det = np.linalg.det(R)
    orth = np.abs(R @ np.transpose(R, (0, 2, 1)) - np.eye(3)).max(axis=(1, 2))
    return {"det_min": float(det.min()), "det_max": float(det.max()), "orth_err_max": float(orth.max())}
