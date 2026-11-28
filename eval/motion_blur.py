"""Measure pose and depth accuracy as more of the ten input views are blurred."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from math import comb
from pathlib import Path
from statistics import fmean

import cv2
import numpy as np
import torch

from evaluate import Dataset, Predictor, REPOSITORY, ROOT
from metrics import crop_depth, frame_depth_metrics, pose_errors, pose_metrics, resize_depth
from vggt_omega.utils.synthetic_blur import (
    DEFAULT_BLUR_PX,
    DEFAULT_REFERENCE_WIDTH,
    blur_source_image,
)


BLUR_COUNTS = (1, 2, 3, 4, 5)
ASSIGNMENTS = 10
DATASETS = ("sintel", "eth3d", "source_val")


def sample_assignments(eligible: list[int], first_seed: int) -> dict[int, dict[int, list[int]]]:
    """Sample distinct blur sets, sharing seeds across counts with the same trial limit."""
    counts_by_limit = {}
    for count in BLUR_COUNTS:
        limit = min(ASSIGNMENTS, comb(len(eligible), count))
        counts_by_limit.setdefault(limit, []).append(count)

    by_count = {}
    for limit, counts in counts_by_limit.items():
        assignments = {}
        seen = {count: set() for count in counts}
        seed = first_seed
        while len(assignments) < limit:
            order = random.Random(seed).sample(eligible, len(eligible))
            subsets = {count: frozenset(order[:count]) for count in counts}
            if all(subset not in seen[count] for count, subset in subsets.items()):
                assignments[seed] = order
                for count, subset in subsets.items():
                    seen[count].add(subset)
            seed += 1
        for count in counts:
            by_count[count] = {seed: order[:count] for seed, order in assignments.items()}
    return by_count


def mean_metrics(rows):
    means = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        means[key] = fmean(values) if values else None
    return means


def write_blurred(source: str, destination: Path, blur_px: float = DEFAULT_BLUR_PX,
                  reference_width: float = DEFAULT_REFERENCE_WIDTH) -> int:
    image = cv2.imread(source, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Unable to read {source}")
    try:
        blurred, kernel_width = blur_source_image(image, blur_px, reference_width, strict=True)
    except ValueError as error:
        raise ValueError(f"Blur did not change {source}") from error
    if not cv2.imwrite(str(destination), blurred):
        raise OSError(f"Unable to save {destination}")
    return kernel_width


def score_views(batch, prediction, blurred_positions):
    depth, poses, crop = prediction
    target = crop_depth(batch.depth, crop)
    depths = list(frame_depth_metrics(resize_depth(depth, target.shape[-2:]), target))
    rotation, translation = pose_errors(poses, batch.extrinsics)
    # Matches the unordered pair ordering in pose_errors (torch.combinations).
    first, second = np.triu_indices(len(poses), k=1)
    blurred = np.isin(np.arange(len(poses)), blurred_positions)
    scores = {}
    for group, mask in (("full", np.ones_like(blurred)), ("clean_only", ~blurred),
                        ("blurred_only", blurred)):
        indices = np.flatnonzero(mask)
        pairs = mask[first] & mask[second]
        rows = [depths[i] for i in indices if depths[i] is not None]
        metrics = {"AUC@3": None, "AUC@30": None, "delta125": None, "AbsRel": None}
        if pairs.any():
            metrics.update(pose_metrics(rotation[pairs], translation[pairs]))
        if rows:
            metrics["delta125"] = float(np.mean([r["delta125"] for r in rows])) * 100
            metrics["AbsRel"] = float(np.mean([r["AbsRel"] for r in rows]))
        elif group == "full":
            raise ValueError("no valid depth frames")
        if any(v is not None and not np.isfinite(v) for v in metrics.values()):
            raise ValueError(f"Non-finite {group} metrics for {batch.frame_names}")
        scores[group] = {"scored_frames": [batch.frame_names[i] for i in indices],
                         "pose_pairs": int(pairs.sum()), "valid_depth_frames": len(rows),
                         "metrics": metrics}
    return scores


def run_scene(args, scene, indices, dataset, predictor):
    batch = dataset.load(scene, indices)
    if len(set(indices)) != 10 or len(indices) != 10:
        raise ValueError("The experiment requires ten distinct benchmark frames")
    positions = list(range(len(indices)))
    assignments = sample_assignments(positions[1:], args.seed)
    selected_positions = {i for group in assignments.values() for mask in group.values() for i in mask}
    scene_dir = args.output_dir / scene
    variants_dir = scene_dir / "blurred"
    variants_dir.mkdir(parents=True, exist_ok=True)
    clean_inputs, crop = predictor.prepare_images(batch.image_paths)
    variant_paths = batch.image_paths.copy()
    fractions = {}
    for i in sorted(selected_positions):
        if args.mixed_severity:
            key = f"mixed-uniform-v1/{args.seed}/{args.dataset}/{scene}/{batch.frame_names[i]}"
            fractions[i] = random.Random(key).uniform(0.02, 0.10)
        else:
            fractions[i] = args.blur_px / args.reference_width
    nominal = {i: f * args.reference_width if args.mixed_severity else args.blur_px for i, f in fractions.items()}
    kernels = {}
    for i in sorted(selected_positions):
        filename = batch.frame_names[i]
        path = variants_dir / Path(filename).with_suffix(".png")
        kernels[filename] = write_blurred(batch.image_paths[i], path, nominal[i],
                                          args.reference_width)
        variant_paths[i] = str(path)
    blur_inputs, blur_crop = predictor.prepare_images(variant_paths)
    if clean_inputs.shape != blur_inputs.shape or crop != blur_crop:
        raise ValueError(f"Blur changed the evaluator input geometry for {scene}")
    runs = {}
    splits = {}

    def run(name, blurred_positions):
        print(f"{scene}/{name}: {len(blurred_positions)} of 10 views blurred", flush=True)
        inputs = torch.stack([blur_inputs[i] if i in blurred_positions else clean_inputs[i]
                              for i in positions])
        prediction = predictor.predict_prepared(inputs, crop)
        runs[name] = {
            "input_frames": batch.frame_names,
            "blurred_frames": [batch.frame_names[i] for i in blurred_positions],
            **score_views(batch, prediction, blurred_positions),
        }

    run("normal", [])
    for count, selections in assignments.items():
        split = f"{len(positions) - count}_{count}"
        trials = []
        for seed, selected in selections.items():
            name = f"blurred_{split}_seed_{seed}"
            run(name, selected)
            trials.append({"seed": seed, "run": name})
        splits[split] = {
            "trials": trials,
            "mean_metrics": mean_metrics([runs[t["run"]]["full"]["metrics"] for t in trials]),
            "subset_metrics": {
                group: mean_metrics([runs[t["run"]][group]["metrics"] for t in trials])
                for group in ("clean_only", "blurred_only")
            },
        }

    result = {
        "dataset": args.dataset, "scene": scene, "seed": args.seed,
        "model_name": args.model_name, "checkpoint": args.checkpoint.name,
        "environment": {"torch": torch.__version__, "opencv": cv2.__version__,
                        "numpy": np.__version__, "device": args.device},
        "blur": {"kernel_xy": [next(iter(kernels.values())), 1] if len(set(kernels.values())) == 1 else None,
                 "source_kernel_widths": kernels, "reference_image_width": args.reference_width,
                 "target_width_fraction": None if args.mixed_severity else args.blur_px / args.reference_width,
                 "width_fraction_by_frame": {batch.frame_names[i]: f for i, f in fractions.items()},
                 "reference_width_by_frame": {batch.frame_names[i]: w for i,w in nominal.items()},
                 "reference_kernel_width": None if args.mixed_severity else args.blur_px,
                 "border": "BORDER_REFLECT_101",
                 "stage": "prepared RGB, before predictor preprocessing"},
        "frames": [
            {"prepared": name, "raw": f"frame_{index + 1:04d}.png" if args.dataset == "sintel" else name}
            for index, name in zip(indices, batch.frame_names)
        ],
        "runs": runs, "splits": splits,
    }
    output = scene_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output.resolve())
    curve = coverage_metrics(result)
    for split, scores in curve.items():
        count = int(split.split("_")[1])
        print(f"{scene} {count / 10:.0%} blurred: AUC@30 {scores['AUC@30']:.2f}, "
              f"AUC@3 {scores['AUC@3']:.2f}, delta125 {scores['delta125']:.2f}, "
              f"AbsRel {scores['AbsRel']:.5f}")
    return {group: coverage_metrics(result, group) for group in ("clean_only", "blurred_only")} | {"full": curve}


def coverage_metrics(result, group="full"):
    curve = {"10_0": result["runs"]["normal"][group]["metrics"]}
    for split, row in result["splits"].items():
        curve[split] = row["mean_metrics"] if group == "full" else row["subset_metrics"][group]
    return curve


def aggregate(curves):
    return {split: mean_metrics([curve[split] for curve in curves]) for split in curves[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="sintel")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPOSITORY / "checkpoint/vggt_omega_1b_416_reproduce.pt")
    parser.add_argument("--model-name", default="original",
                        help="label for this checkpoint in outputs/blur_study/<dataset>/<model-name>")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an existing, non-empty output directory first")
    parser.add_argument("--mixed-severity", action="store_true",
                        help="give each blurred frame its own width, uniform in 2-10%% of the image "
                             "width and seeded by frame name (the mixed_uniform condition); "
                             "--blur-px is then ignored")
    parser.add_argument("--blur-px", type=float, default=DEFAULT_BLUR_PX,
                        help="kernel width at the reference width (study severity: 31)")
    parser.add_argument("--reference-width", type=float, default=DEFAULT_REFERENCE_WIDTH)
    parser.add_argument("--frames", type=Path, help="frame map JSON (default eval/frames/<dataset>.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.frames = args.frames or ROOT / "frames" / f"{args.dataset}.json"
    frame_map = json.loads(args.frames.read_text())
    args.scenes = args.scenes or list(frame_map)
    unknown = set(args.scenes).difference(frame_map)
    if unknown:
        parser.error(f"Unknown {args.dataset} scenes: {sorted(unknown)}")
    if args.output_dir is None:
        args.output_dir = REPOSITORY / "outputs/blur_study" / args.dataset / args.model_name
    args.data_root = args.data_root or REPOSITORY / "data" / args.dataset
    dataset = Dataset(args.data_root)
    predictor = Predictor(args.checkpoint, args.device)
    output_dir = args.output_dir.resolve()
    protected = (Path.cwd(), REPOSITORY, args.data_root.resolve(), args.checkpoint.resolve())
    if (Path(os.path.abspath(args.output_dir)) != output_dir
            or any(path.is_relative_to(output_dir) for path in protected)
            or output_dir.is_relative_to(args.data_root.resolve())
            or output_dir.is_relative_to(ROOT)):
        parser.error(f"Unsafe output directory: {args.output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            parser.error(f"{output_dir} is not empty; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    per_scene = {
        scene: run_scene(args, scene, frame_map[scene], dataset, predictor)
        for scene in dict.fromkeys(args.scenes)
    }
    report = {}
    for group in ("full", "clean_only", "blurred_only"):
        curves = {scene: scores[group] for scene, scores in per_scene.items()}
        summary = aggregate(list(curves.values()))
        summary["per_scene"] = curves
        if group == "full":
            report.update(summary)
        else:
            report[group] = summary
    report["meta"] = {
        "dataset": args.dataset, "model_name": args.model_name,
        "checkpoint": str(args.checkpoint), "seed": args.seed,
        "blur_px": None if args.mixed_severity else args.blur_px, "reference_width": args.reference_width,
        "mixed_distribution": "uniform_0.02_0.10_per_frame_v1" if args.mixed_severity else None,
        "scenes": list(per_scene), "frames": str(args.frames),
        "aggregation": "placements averaged within each scene, then scenes weighted equally",
    }
    output = args.output_dir / "coverage_metrics.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
