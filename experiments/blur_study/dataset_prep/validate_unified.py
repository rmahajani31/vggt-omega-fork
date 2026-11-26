"""Validate a converted unified dataset (structural, geometric and loader checks).

Per download_and_convert.md:
  * structural: every scene through unified_io.validate_scene (fast or --full);
  * coverage: every expected manifest scene present, nothing extra;
  * geometry: depth reprojection between nearby frames, compared against wrong-hypothesis
    controls on the same pairs (pose inverted, focal +10%, one-frame association shift);
  * loader: the real UnifiedDataset.get_data with explicit seq_name / ids, single-frame and
    ten-frame, followed by ComposedDataset post-processing (the training path).
Fails closed: any exception or failed scene makes the exit status non-zero.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from convert_common import MANIFEST, REPOSITORY, UNIFIED, manifest_rows
from geometry import reproject_score
from unified_io import validate_scene

LABELS = {"uco3d": "uCO3D", "hypersim": "Hypersim", "mvssynth": "MVS-Synth"}


def load_scene_arrays(scene_dir: Path):
    import cv2

    names = json.loads((scene_dir / "image_names.json").read_text())
    poses = np.load(scene_dir / "cam_from_worlds.npy").astype(np.float64)
    K = np.load(scene_dir / "intrinsics.npy").astype(np.float64)

    def depth(i):
        return cv2.imread(str(scene_dir / "depths" / f"{Path(names[i]).stem}.exr"), cv2.IMREAD_UNCHANGED)

    return names, poses, K, depth


def geometry_check(scene_dir: Path, pairs: int, gap: int, rng: random.Random) -> dict:
    names, poses, K, depth = load_scene_arrays(scene_dir)
    n = len(names)
    if n < gap + 2:
        return {"pairs": 0}
    starts = sorted(rng.sample(range(1, n - gap), min(pairs, n - gap - 1)))
    rows = {k: [] for k in ("documented", "pose_inverted", "focal_plus_10pct", "frame_shift_plus_1")}
    for i in starts:
        j = i + gap
        di, dj, dj_shift = depth(i), depth(j), depth(min(j + 1, n - 1))
        Kf = K[i].copy()
        Kf[0, 0] *= 1.1
        Kf[1, 1] *= 1.1
        inv = lambda P: np.linalg.inv(np.vstack([P, [0, 0, 0, 1]]))[:3]  # noqa: E731
        cases = {
            "documented": (di, K[i], poses[i], dj, K[j], poses[j]),
            "pose_inverted": (di, K[i], inv(poses[i]), dj, K[j], inv(poses[j])),
            "focal_plus_10pct": (di, Kf, poses[i], dj, Kf, poses[j]),
            "frame_shift_plus_1": (di, K[i], poses[i], dj_shift, K[j], poses[j]),
        }
        for name, case in cases.items():
            s = reproject_score(*case)
            rows[name].append((s.overlap, s.inlier, s.median_rel))
    summary = {}
    for name, values in rows.items():
        v = np.array(values, dtype=np.float64)
        summary[name] = {"overlap": float(np.nanmean(v[:, 0])), "inlier_1pct": float(np.nanmean(v[:, 1])),
                         "median_rel": float(np.nanmedian(v[:, 2]))}
    summary["pairs"] = len(starts)
    summary["gap"] = gap
    return summary


def loader_probe(dataset_key: str, root: Path, scenes: list[str], seeds=(0, 1)) -> dict:
    import os

    os.chdir(REPOSITORY / "training")
    sys.path.insert(0, str(REPOSITORY / "training"))
    from hydra.utils import instantiate
    from launch import load_config

    cfg = load_config("blur_study")
    conf = cfg.data.train.dataset.dataset_configs[dataset_key]
    conf.UNIFIED_DIR = str(root)
    conf.sequence_list_file = None
    common = cfg.data.train.common_config
    from data.composed_dataset import ComposedDataset

    # The real training composition (augmentation, tracks, normalisation) around one dataset.
    post = ComposedDataset({dataset_key: conf}, common)
    dataset = post.datasets_by_key[dataset_key]
    out = {"single_frame": 0, "multi_frame": 0, "errors": []}
    for scene in scenes:
        try:
            names = dataset.load_image_names(scene)
            for idx in (0, len(names) // 2, len(names) - 1):
                batch = dataset.get_data(seq_name=scene, ids=[idx], img_per_seq=1, exact_ids=True,
                                         aspect_ratio=1.0)
                if not (np.isfinite(batch["depths"][0]).all() and (batch["depths"][0] > 0).any()):
                    raise ValueError(f"{scene}[{idx}]: no valid loader depth")
                out["single_frame"] += 1
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                ids = dataset.sample_frame_ids(scene, 10)
                batch = dataset.get_data(seq_name=scene, ids=ids, img_per_seq=10, exact_ids=True,
                                         aspect_ratio=0.75)
                sample = post.postprocess(batch)
                for key in ("images", "depths", "extrinsics", "intrinsics", "world_points"):
                    if not torch_finite(sample[key]):
                        raise ValueError(f"{scene}: non-finite {key} after post-processing")
                if sample["images"].shape[0] != 10:
                    raise ValueError(f"{scene}: wrong view count")
                out["multi_frame"] += 1
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"{scene}: {type(e).__name__}: {e}")
    return out


def torch_finite(x) -> bool:
    import torch

    return bool(torch.isfinite(x).all()) if torch.is_tensor(x) else bool(np.isfinite(x).all())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(LABELS))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--full", action="store_true", help="decode every frame (slow)")
    parser.add_argument("--allow-missing", action="store_true", help="partial conversions")
    parser.add_argument("--geometry-scenes", type=int, default=12)
    parser.add_argument("--geometry-pairs", type=int, default=4)
    parser.add_argument("--gap", type=int, default=4,
                        help="frame separation of geometry pairs (adjacent frames barely test focal errors)")
    parser.add_argument("--loader-scenes", type=int, default=4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = (args.root or UNIFIED / args.dataset).resolve()
    # Resolve now: the loader probe changes the working directory to training/.
    report_path = args.report.resolve() if args.report else root.parent / "_reports" / f"{args.dataset}_validation.json"
    expected = {r["catalog_id"] for r in manifest_rows(LABELS[args.dataset], args.manifest)}
    present = {p.name for p in root.iterdir() if p.is_dir()}
    report = {"dataset": args.dataset, "root": str(root), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "expected": len(expected), "present": len(present),
              "missing": sorted(expected - present), "extra": sorted(present - expected)}
    failed = []
    if report["extra"] or (report["missing"] and not args.allow_missing):
        failed.append("coverage")

    structural = {}
    for scene in sorted(present & expected):
        try:
            structural[scene] = validate_scene(root / scene, full=args.full)
        except Exception as e:  # noqa: BLE001
            structural[scene] = {"errors": [f"{type(e).__name__}: {e}"], "warnings": []}
    bad = {s: v["errors"] for s, v in structural.items() if v["errors"]}
    if bad:
        failed.append("structural")
    report["structural"] = {
        "scenes": len(structural), "failed": bad,
        "frames": int(sum(v.get("frames", 0) for v in structural.values())),
        "warnings": {s: v["warnings"] for s, v in structural.items() if v.get("warnings")},
        "depth_coverage_mean": float(np.mean([v.get("depth_coverage_mean", 0) for v in structural.values()]))
        if structural else 0.0,
    }

    rng = random.Random(0)
    good = sorted(s for s in structural if s not in bad)
    geo_scenes = rng.sample(good, min(args.geometry_scenes, len(good)))
    geometry = {s: geometry_check(root / s, args.geometry_pairs, args.gap, rng) for s in geo_scenes}
    report["geometry"] = geometry
    agg = {}
    for result in geometry.values():
        for name, v in result.items():
            if isinstance(v, dict):
                agg.setdefault(name, []).append(v["inlier_1pct"])
    report["geometry_summary"] = {k: float(np.nanmean(v)) for k, v in agg.items()}
    doc = report["geometry_summary"].get("documented", 0.0)
    controls = [v for k, v in report["geometry_summary"].items() if k != "documented"]
    if controls and not doc > 2 * max(controls):
        failed.append("geometry (documented convention does not clearly beat the controls)")

    probe = loader_probe(args.dataset, root, rng.sample(good, min(args.loader_scenes, len(good))))
    report["loader_probe"] = probe
    if probe["errors"]:
        failed.append("loader")
    report["failed_checks"] = failed
    report["status"] = "pass" if not failed else "fail"

    out = report_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dataset", "expected", "present", "status", "failed_checks",
                                             "geometry_summary")}, indent=1))
    print(f"structural: {report['structural']['scenes']} scenes, {report['structural']['frames']} frames, "
          f"{len(bad)} failed; loader probe: {probe['single_frame']} single, {probe['multi_frame']} multi, "
          f"{len(probe['errors'])} errors; report {out}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
