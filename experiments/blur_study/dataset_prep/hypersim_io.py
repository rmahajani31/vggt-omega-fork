"""Hypersim raw readers and the documented Hypersim -> OpenCV conversions.

Source conventions (ml-hypersim README and contrib/mikeroberts3000 camera notebook):
  * ``camera_keyframe_orientations`` holds R_world_from_cam (columns are camera axes in world);
    the Hypersim camera has x right, y up, z backwards (it looks along -z).
  * ``camera_keyframe_positions`` holds camera centres in asset units; metres =
    asset units * ``meters_per_asset_unit`` (``_detail/metadata_scene.csv``).
  * ``frame.XXXX.depth_meters.hdf5`` is the Euclidean distance (metres) from the optical centre,
    not planar depth; NaN marks pixels with no geometry.
  * ``M_cam_from_uv`` (``metadata_camera_parameters.csv``) maps normalised image coords
    u, v in [-1, 1] (u right, v up; u = 2 (col + 0.5) / W - 1, v = 1 - 2 (row + 0.5) / H) to a
    camera-space ray d = M @ [u, v, 1]. Any invertible M is a central (pinhole) projection; for
    the study's 140 scenes M is diagonal (85), has a lens shift (M_02/M_12), or is a tilt-shift
    camera with M_20/M_21 != 0 (52), i.e. an image plane rotated by up to ~11 degrees.

Conversion to the unified (OpenCV, zero-skew pinhole) contract:
  A = D @ M @ N maps pixel centres p = (col + 0.5, row + 0.5, 1) to OpenCV-camera rays, with
  D = diag(1, -1, -1) (Hypersim -> OpenCV axes) and N the pixel -> (u, v) affine map.
  RQ-decompose A^-1 = K_s R_t (K_s upper triangular with positive diagonal, R_t a rotation).
  The OpenCV camera is rotated by R_t: cam_from_world = R_t @ [D R^T | -D R^T p_m].
  If K_s has (numerically) zero skew the pixels already form that pinhole image and are kept;
  otherwise RGB and depth are resampled with the homography K @ K_s^-1 onto K = K_s with the
  skew removed (only scenes whose tilt has both a horizontal and a vertical component).
  Planar depth in the new camera: z = range * (R_t a)_z / ||a||, a = A p.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

D = np.diag([1.0, -1.0, -1.0])


@lru_cache(maxsize=4)
def camera_parameters(csv_path: str) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["scene_name"]: row for row in csv.DictReader(f)}


def meters_per_asset_unit(scene_dir: Path) -> float:
    with open(scene_dir / "_detail" / "metadata_scene.csv", newline="", encoding="utf-8") as f:
        rows = {r["parameter_name"]: r["parameter_value"] for r in csv.DictReader(f)}
    return float(rows["meters_per_asset_unit"])


SKEW_TOLERANCE_PX = 1e-3


@dataclass
class HypersimCamera:
    K: np.ndarray            # 3x3 zero-skew OpenCV intrinsics for the stored image
    R_tilt: np.ndarray       # rotation applied in the camera frame (identity unless tilted)
    A: np.ndarray            # source pixel centre -> OpenCV ray (before R_tilt)
    warp: np.ndarray | None  # source -> stored pixel homography, None when pixels are kept
    height: int
    width: int


def opencv_camera(row: dict) -> HypersimCamera:
    from scipy.linalg import rq

    W = int(float(row["settings_output_img_width"]))
    H = int(float(row["settings_output_img_height"]))
    M = np.array([[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)])
    N = np.array([[2.0 / W, 0, -1.0], [0, -2.0 / H, 1.0], [0, 0, 1.0]])
    A = D @ M @ N
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError(f"Degenerate M_cam_from_uv: {M.tolist()}")
    K_s, R_t = rq(np.linalg.inv(A))
    signs = np.diag(np.sign(np.diag(K_s)))
    K_s, R_t = K_s @ signs, signs @ R_t
    if np.linalg.det(R_t) < 0:
        K_s, R_t = -K_s, -R_t
    K_s = K_s / K_s[2, 2]
    if K_s[0, 0] <= 0 or K_s[1, 1] <= 0:
        raise ValueError(f"Non-positive focal length from M_cam_from_uv: {M.tolist()}")
    K = K_s.copy()
    K[0, 1] = 0.0
    warp = None if abs(K_s[0, 1]) < SKEW_TOLERANCE_PX else K @ np.linalg.inv(K_s)
    if not (0 < K[0, 2] < W and 0 < K[1, 2] < H):
        raise ValueError(f"Principal point outside the image: {K.tolist()}")
    return HypersimCamera(K=K, R_tilt=R_t, A=A, warp=warp, height=H, width=W)


def intrinsics_from_row(row: dict) -> tuple[np.ndarray, int, int]:
    cam = opencv_camera(row)
    return cam.K, cam.height, cam.width


def cam_from_world(R_world_from_cam: np.ndarray, position_m: np.ndarray,
                   R_tilt: np.ndarray | None = None) -> np.ndarray:
    R = D @ R_world_from_cam.T
    if R_tilt is not None:
        R = R_tilt @ R
    return np.concatenate([R, (-R @ position_m)[:, None]], axis=1)


def planar_depth(range_map: np.ndarray, camera: HypersimCamera) -> np.ndarray:
    """Range (metres along the ray) -> z in the rotated OpenCV camera, at source pixels."""
    H, W = range_map.shape
    rows, cols = np.mgrid[0:H, 0:W]
    pixels = np.stack([cols + 0.5, rows + 0.5, np.ones_like(cols, dtype=np.float64)], -1)
    rays = pixels @ camera.A.T
    z_dir = (rays @ camera.R_tilt.T)[..., 2] / np.linalg.norm(rays, axis=-1)
    z = range_map * z_dir
    return np.where(np.isfinite(z) & (z > 0), z, 0).astype(np.float32)


def warp_to_pinhole(image: np.ndarray, depth: np.ndarray, camera: HypersimCamera):
    """Resample skewed scenes onto the zero-skew grid (cubic RGB, nearest depth)."""
    import cv2

    if camera.warp is None:
        return image, depth
    size = (camera.width, camera.height)
    # warpPerspective works on integer pixel coordinates; shift by the half-pixel centre.
    shift = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    Hm = np.linalg.inv(shift) @ camera.warp @ shift
    image = cv2.warpPerspective(image, Hm, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    depth = cv2.warpPerspective(depth, Hm, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return image, depth


def load_trajectory(scene_dir: Path, camera: str):
    import h5py

    detail = scene_dir / "_detail" / camera
    with h5py.File(detail / "camera_keyframe_frame_indices.hdf5", "r") as f:
        frame_indices = f["dataset"][:].astype(np.int64)
    with h5py.File(detail / "camera_keyframe_orientations.hdf5", "r") as f:
        orientations = f["dataset"][:].astype(np.float64)
    with h5py.File(detail / "camera_keyframe_positions.hdf5", "r") as f:
        positions = f["dataset"][:].astype(np.float64)
    if not (len(frame_indices) == len(orientations) == len(positions)):
        raise ValueError(f"Keyframe arrays disagree in {detail}")
    return frame_indices, orientations, positions


def read_range(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        return f["dataset"][:].astype(np.float32)


def range_to_z(range_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = range_map.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u + 0.5 - K[0, 2]) / K[0, 0]
    y = (v + 0.5 - K[1, 2]) / K[1, 1]
    z = range_map / np.sqrt(x * x + y * y + 1.0)
    return np.where(np.isfinite(z) & (z > 0), z, 0).astype(np.float32)


def frame_records(scene_dir: Path, camera: str):
    """Frames with RGB, depth and a keyframe pose, in frame order: (frame_id, jpg, hdf5, k)."""
    frame_indices, _, _ = load_trajectory(scene_dir, camera)
    keyframe_of = {int(f): k for k, f in enumerate(frame_indices)}
    rgb_dir = scene_dir / "images" / f"scene_{camera}_final_preview"
    geo_dir = scene_dir / "images" / f"scene_{camera}_geometry_hdf5"
    records, skipped = [], []
    for jpg in sorted(rgb_dir.glob("frame.*.tonemap.jpg")):
        frame_id = int(jpg.name.split(".")[1])
        depth = geo_dir / f"frame.{frame_id:04d}.depth_meters.hdf5"
        if not depth.is_file():
            skipped.append((frame_id, "missing depth_meters"))
        elif frame_id not in keyframe_of:
            skipped.append((frame_id, "no camera keyframe"))
        else:
            records.append((frame_id, jpg, depth, keyframe_of[frame_id]))
    return records, skipped
