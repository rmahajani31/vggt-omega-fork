from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def validate_scene(scene_dir: Path, full: bool = True) -> dict:
    """Structural checks from the guide. Returns {"errors": [...], "warnings": [...], stats}."""
    scene_dir = Path(scene_dir)
    errors, warnings = [], []
    try:
        names = json.loads((scene_dir / "image_names.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"errors": [f"image_names.json: {e}"], "warnings": []}
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        errors.append("image_names.json is not a list of strings")
        return {"errors": errors, "warnings": warnings}
    stems = [Path(n).stem for n in names]
    if len(set(names)) != len(names) or len(set(stems)) != len(stems):
        errors.append("duplicate image names or stems")
    if any("/" in n or "\\" in n or n.startswith(".") for n in names):
        errors.append("unsafe image names")
    poses = np.load(scene_dir / "cam_from_worlds.npy")
    K = np.load(scene_dir / "intrinsics.npy")
    n = len(names)
    if poses.shape != (n, 3, 4):
        errors.append(f"cam_from_worlds shape {poses.shape} != ({n}, 3, 4)")
    if K.shape != (n, 3, 3):
        errors.append(f"intrinsics shape {K.shape} != ({n}, 3, 3)")
    if errors:
        return {"errors": errors, "warnings": warnings}
    if not (np.isfinite(poses).all() and np.isfinite(K).all()):
        errors.append("non-finite poses or intrinsics")
    R = poses[:, :, :3].astype(np.float64)
    det = np.linalg.det(R)
    orth = np.abs(R @ np.transpose(R, (0, 2, 1)) - np.eye(3)).max(axis=(1, 2))
    if np.abs(det - 1).max() > 1e-3 or orth.max() > 1e-3:
        errors.append(f"improper rotations: det in [{det.min():.4f}, {det.max():.4f}], orth err {orth.max():.2e}")
    if (K[:, 0, 0] <= 0).any() or (K[:, 1, 1] <= 0).any():
        errors.append("non-positive focal length")
    if np.abs(K[:, 0, 1]).max() > 1e-6 or np.abs(K[:, 1, 0]).max() > 0 or np.abs(K[:, 2, :2]).max() > 0 \
            or np.abs(K[:, 2, 2] - 1).max() > 0:
        errors.append("intrinsics are not [[fx,0,cx],[0,fy,cy],[0,0,1]]")
    if K.min() < 0:
        errors.append("negative intrinsic entries (rejected by the loader)")

    image_files = {p.name for p in (scene_dir / "images").iterdir()}
    depth_files = {p.name for p in (scene_dir / "depths").iterdir()}
    if image_files != set(names):
        errors.append(f"orphan/missing images: {sorted(image_files ^ set(names))[:5]}")
    if depth_files != {f"{s}.exr" for s in stems}:
        errors.append(f"orphan/missing depths: {sorted(depth_files ^ {f'{s}.exr' for s in stems})[:5]}")

    coverage, zero_frames = [], 0
    check = range(n) if full else sorted({0, n // 2, n - 1})
    for i in check:
        image = cv2.imread(str(scene_dir / "images" / names[i]), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(scene_dir / "depths" / f"{stems[i]}.exr"), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            errors.append(f"{names[i]}: image does not decode as HxWx3 uint8")
            continue
        if depth is None or depth.ndim != 2 or depth.dtype != np.float32:
            errors.append(f"{names[i]}: depth is not a 2D float32 EXR")
            continue
        if depth.shape != image.shape[:2]:
            errors.append(f"{names[i]}: depth {depth.shape} != image {image.shape[:2]}")
        if not np.isfinite(depth).all() or depth.min() < 0:
            errors.append(f"{names[i]}: non-finite or negative depth")
        h, w = depth.shape
        if not (0 < K[i, 0, 2] < w and 0 < K[i, 1, 2] < h):
            errors.append(f"{names[i]}: principal point outside the image")
        valid = float((depth > 0).mean())
        coverage.append(valid)
        zero_frames += valid == 0
    if zero_frames:
        warnings.append(f"{zero_frames} frames have no valid depth")
    if n < 3:
        warnings.append(f"only {n} frames")
    return {
        "errors": errors, "warnings": warnings, "frames": n, "checked_frames": len(check),
        "depth_coverage_mean": float(np.mean(coverage)) if coverage else 0.0,
        "depth_coverage_min": float(np.min(coverage)) if coverage else 0.0,
        "det_range": [float(det.min()), float(det.max())],
    }
