"""Build the 50-recording source-validation cohort in the evaluator's scene layout.

For every ``source_val50`` row of the study manifest, ten distinct views are drawn with that
dataset's native UnifiedDataset sampler (the blur_study loader settings) under a fixed seed, so
validation samples look like training samples. The ten frames are exported, in sampled order,
to ``<out>/<dataset>__<recording>/`` with the layout eval/datasets.py reads:

    frames.txt, cameras.npz (extrinsics N x 3 x 4, intrinsics N x 3 x 3),
    images/<name>, depths/<stem>.npz (key "depth")

and the frame map is written to eval/frames/source_val.json (indices 0..9 per scene). The
manifest's Sintel and ETH3D frame indices are also checked against eval/frames/*.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
TRAINING = REPOSITORY / "training"
for path in (TRAINING, REPOSITORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402

from data.blur_schedule import manifest_recordings, read_manifest, stable_seed  # noqa: E402

VIEWS = 10
MAX_ATTEMPTS = 100


def check_benchmark_frames(rows):
    for cohort, name in (("sintel", "sintel"), ("eth3d", "eth3d")):
        frame_map = json.loads((ROOT / "frames" / f"{name}.json").read_text())
        for row in rows:
            if row["cohort"] != cohort:
                continue
            expected = json.loads(row["eval_frame_indices_0based"])
            if frame_map.get(row["sequence_id"]) != expected:
                raise SystemExit(f"Manifest {cohort}/{row['sequence_id']} frames {expected} "
                                 f"!= eval/frames/{name}.json {frame_map.get(row['sequence_id'])}")
    print("Manifest Sintel/ETH3D frame indices match eval/frames/*.json")


def draw_distinct(dataset, seq_name, seed, image_names):
    import random
    for attempt in range(MAX_ATTEMPTS):
        s = stable_seed(seed, "source_val", seq_name, attempt)
        random.seed(s)
        np.random.seed(s)
        ids = [int(i) for i in dataset.sample_frame_ids(seq_name, VIEWS, image_names=image_names)]
        if len(set(ids)) == VIEWS:
            return ids, attempt
    raise RuntimeError(f"Could not draw {VIEWS} distinct views from {seq_name}")


def export_scene(dataset, seq_name, ids, image_names, destination: Path):
    source = Path(dataset.UNIFIED_DIR) / seq_name
    cam_from_worlds = np.load(source / dataset.cam_from_worlds_filename)
    intrinsics = np.load(source / dataset.intrinsics_filename)
    tmp = destination.with_name(destination.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "images").mkdir(parents=True)
    (tmp / "depths").mkdir()
    names = []
    for idx in ids:
        name = image_names[idx]
        shutil.copy2(source / "images" / name, tmp / "images" / name)
        depth = cv2.imread(str(source / "depths" / (Path(name).stem + ".exr")), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            raise ValueError(f"Bad depth for {seq_name}/{name}")
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        np.savez_compressed(tmp / "depths" / (Path(name).stem + ".npz"), depth=depth)
        names.append(name)
    (tmp / "frames.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    np.savez(tmp / "cameras.npz", extrinsics=cam_from_worlds[ids].astype(np.float64),
             intrinsics=intrinsics[ids].astype(np.float64))
    if destination.exists():
        shutil.rmtree(destination)
    tmp.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPOSITORY / "data/vggt_omega_train_validation_v2.csv",
                        help="study manifest (v2: uCO3D rows missing from the authors upload replaced)")
    parser.add_argument("--out", type=Path, default=REPOSITORY / "data/source_val")
    parser.add_argument("--frames-out", type=Path, default=ROOT / "frames/source_val.json")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--check-only", action="store_true", help="only check Sintel/ETH3D frame maps")
    parser.add_argument("overrides", nargs="*", help="hydra overrides for the blur_study config")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    check_benchmark_frames(rows)
    if args.check_only:
        return

    manifest, out, frames_out = args.manifest.resolve(), args.out.resolve(), args.frames_out.resolve()
    os.chdir(TRAINING)
    from hydra.utils import instantiate
    from launch import load_config

    cfg = load_config("blur_study", args.overrides)
    common = cfg.data.train.common_config
    configs = cfg.data.train.dataset.dataset_configs
    recordings = manifest_recordings(rows, "validation", cohort="source_val50")
    datasets = {}
    for key in sorted({r["dataset"] for r in recordings}):
        conf = dict(configs[key])
        conf["sequence_list_file"] = None  # validation scenes are not in the training lists
        datasets[key] = instantiate(conf, common_conf=common)

    frame_map, selection = {}, []
    out.mkdir(parents=True, exist_ok=True)
    for rec in recordings:
        dataset = datasets[rec["dataset"]]
        image_names = dataset.load_image_names(rec["seq_name"])
        ids, attempt = draw_distinct(dataset, rec["seq_name"], args.seed, image_names)
        # Short, unique scene names (uCO3D ids drop "_rgb_video_<md5>") keep eval output paths
        # under the Windows path limit; selection.json maps each back to its catalog id.
        scene = f"{rec['dataset']}__{rec['seq_name'].split('_rgb_video_')[0]}"
        export_scene(dataset, rec["seq_name"], ids, image_names, out / scene)
        frame_map[scene] = list(range(VIEWS))
        selection.append({"scene": scene, "dataset": rec["dataset"], "catalog_id": rec["seq_name"],
                          "source_frame_ids": ids, "frame_names": [image_names[i] for i in ids],
                          "draw_attempt": attempt})
        print(f"{scene}: frames {ids}")

    frames_out.parent.mkdir(parents=True, exist_ok=True)
    frames_out.write_text(json.dumps(frame_map, indent=1) + "\n", encoding="utf-8")
    (out / "selection.json").write_text(json.dumps(
        {"seed": args.seed, "manifest": str(manifest), "scenes": selection}, indent=1) + "\n",
        encoding="utf-8")
    # Filled-in copy of the manifest; the original CSV is left untouched.
    by_id = {(s["dataset"], s["catalog_id"]): s for s in selection}
    filled = out / "manifest_with_source_val_frames.csv"
    with open(filled, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        from data.blur_schedule import MANIFEST_DATASETS
        for row in rows:
            key = (MANIFEST_DATASETS.get(row["dataset"]), row["catalog_id"])
            if row["cohort"] == "source_val50" and key in by_id:
                row = dict(row)
                row["eval_frame_indices_0based"] = json.dumps(by_id[key]["source_frame_ids"])
                row["eval_frame_names"] = json.dumps(by_id[key]["frame_names"])
            writer.writerow(row)
    print(f"Wrote {len(selection)} scenes to {out} and {frames_out}")


if __name__ == "__main__":
    main()
