"""Write and validate scenes in the VGGT-Omega unified layout (training/README.md section 2).

Implements the contract in training/dataset_preparation/download_and_convert.md:
  * one directory per scene with images/, depths/<stem>.exr, image_names.json,
    cam_from_worlds.npy (N x 3 x 4, OpenCV camera-from-world) and intrinsics.npy (N x 3 x 3);
  * depth is float32 planar z in metres, every invalid value stored as exactly 0;
  * each scene is staged outside the dataset root, validated there and published with an
    atomic rename; a completion record (converter version, config hash, counts, validation)
    is written last, outside the dataset root, so reruns resume safely.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402


@dataclass
class Frame:
    name: str                      # output image file name, e.g. 00000.png
    cam_from_world: np.ndarray     # 3x4 (or 4x4), OpenCV
    K: np.ndarray                  # 3x3
    depth: np.ndarray              # HxW float, invalid <= 0 / non-finite
    image_path: Path | None = None  # copied verbatim when given (no re-encode)
    image: np.ndarray | None = None  # RGB uint8 HxWx3, encoded losslessly when given
    source_id: str = ""


@dataclass
class SceneResult:
    scene: str
    status: str                    # converted | skipped | failed
    frames: int = 0
    skipped_frames: list = field(default_factory=list)
    error: str = ""
    validation: dict = field(default_factory=dict)


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def exr_roundtrip_probe(tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    probe = np.array([[0.0, 1.5], [1234.25, 0.001]], dtype=np.float32)
    path = tmp_dir / "probe.exr"
    if not cv2.imwrite(str(path), probe):
        raise RuntimeError("OpenCV cannot write EXR; set OPENCV_IO_ENABLE_OPENEXR=1 before importing cv2")
    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    path.unlink()
    if back is None or back.dtype != np.float32 or not np.array_equal(back, probe):
        raise RuntimeError("EXR float32 round-trip failed")


def clean_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)


def record_path(records_root: Path, scene: str) -> Path:
    return records_root / f"{scene}.json"


def is_complete(dest_root: Path, records_root: Path, scene: str, version: str, cfg_hash: str) -> bool:
    record = record_path(records_root, scene)
    if not record.is_file() or not (dest_root / scene / "image_names.json").is_file():
        return False
    data = json.loads(record.read_text(encoding="utf-8"))
    return (data.get("status") == "converted" and data.get("converter_version") == version
            and data.get("config_hash") == cfg_hash)


def write_scene(dest_root: Path, records_root: Path, scene: str, frames: Iterable[Frame],
                *, version: str, cfg_hash: str, provenance: dict, overwrite: bool = False) -> SceneResult:
    dest_root, records_root = Path(dest_root), Path(records_root)
    final = dest_root / scene
    if final.exists() and not overwrite:
        if is_complete(dest_root, records_root, scene, version, cfg_hash):
            return SceneResult(scene, "skipped")
        raise FileExistsError(f"{final} exists without a matching completion record; use --overwrite")
    staging_root = dest_root.parent / "_staging" / dest_root.name
    staging = staging_root / f"{scene}.{os.getpid()}.{int(time.time())}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)
    (staging / "depths").mkdir()

    names, poses, intrinsics, sources = [], [], [], []
    for frame in frames:
        stem = Path(frame.name).stem
        if frame.image_path is not None:
            shutil.copyfile(frame.image_path, staging / "images" / frame.name)
        elif frame.image is not None:
            ok = cv2.imwrite(str(staging / "images" / frame.name), frame.image[:, :, ::-1])
            if not ok:
                raise OSError(f"Failed to write {frame.name}")
        else:
            raise ValueError(f"Frame {frame.name} has no image")
        depth = clean_depth(frame.depth)
        if not cv2.imwrite(str(staging / "depths" / f"{stem}.exr"), depth):
            raise OSError(f"Failed to write depth for {frame.name}")
        back = cv2.imread(str(staging / "depths" / f"{stem}.exr"), cv2.IMREAD_UNCHANGED)
        if back is None or not np.array_equal(back, depth):
            raise OSError(f"Depth round-trip failed for {frame.name}")
        names.append(frame.name)
        poses.append(np.asarray(frame.cam_from_world, dtype=np.float64)[:3, :4])
        intrinsics.append(np.asarray(frame.K, dtype=np.float64))
        sources.append(frame.source_id)
    if not names:
        shutil.rmtree(staging)
        raise ValueError(f"Scene {scene} has no frames")
    (staging / "image_names.json").write_text(json.dumps(names), encoding="utf-8")
    np.save(staging / "cam_from_worlds.npy", np.stack(poses).astype(np.float32))
    np.save(staging / "intrinsics.npy", np.stack(intrinsics).astype(np.float32))

    validation = validate_scene(staging)
    if validation["errors"]:
        raise ValueError(f"Staged scene {scene} failed validation: {validation['errors'][:5]}")

    backup = None
    if final.exists():
        backup = final.with_name(final.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        final.rename(backup)
    dest_root.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    if backup is not None:
        shutil.rmtree(backup)
    records_root.mkdir(parents=True, exist_ok=True)
    record = {
        "scene": scene, "status": "converted", "converter_version": version, "config_hash": cfg_hash,
        "frames": len(names), "source_frames": sources, "provenance": provenance,
        "validation": validation, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    record_path(records_root, scene).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return SceneResult(scene, "converted", frames=len(names), validation=validation)


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
