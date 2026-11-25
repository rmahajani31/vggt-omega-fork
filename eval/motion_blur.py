"""Measure pose and depth accuracy as more of the ten input views are blurred."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from statistics import fmean

import cv2
import numpy as np
import torch

from evaluate import Dataset, Predictor, REPOSITORY, ROOT, pose_metrics, score_scene


ORIGINAL_COUNTS = (2, 3, 4, 5)
ASSIGNMENTS = 10
KERNEL = (61, 1)
SINTEL_IMAGE_WIDTH = 846
BLUR_WIDTH_FRACTION = KERNEL[0] / SINTEL_IMAGE_WIDTH
RESULT_FOLDERS = {"sintel": "sintel_sequences_61px", "eth3d": "eth3d_sequences_61px_width_matched"}


def sample_assignments(eligible: list[int], first_seed: int) -> dict[int, dict[int, list[int]]]:
    """Return distinct sets per ratio, retaining the original four-ratio cohort."""
    assignments = {}
    seen = {count: set() for count in ORIGINAL_COUNTS}
    seed = first_seed
    while len(assignments) < ASSIGNMENTS:
        order = random.Random(seed).sample(eligible, len(eligible))
        subsets = {count: frozenset(order[:count]) for count in ORIGINAL_COUNTS}
        if all(subset not in seen[count] for count, subset in subsets.items()):
            assignments[seed] = order
            for count, subset in subsets.items():
                seen[count].add(subset)
        seed += 1
    by_count = {
        count: {seed: order[:count] for seed, order in assignments.items()}
        for count in ORIGINAL_COUNTS
    }
    return by_count


def mean_metrics(rows):
    return {key: fmean(row[key] for row in rows) for key in rows[0]}


def write_blurred(source: str, destination: Path) -> int:
    image = cv2.imread(source, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Unable to read {source}")
    source_width = BLUR_WIDTH_FRACTION * image.shape[1]
    kernel_width = max(1, 2 * round((source_width - 1) / 2) + 1)
    blurred = cv2.blur(image, (kernel_width, 1), borderType=cv2.BORDER_REFLECT_101)
    if np.array_equal(image, blurred):
        raise ValueError(f"Blur did not change {source}")
    if not cv2.imwrite(str(destination), blurred):
        raise OSError(f"Unable to save {destination}")
    return kernel_width


def score_views(batch, prediction):
    depth, poses, crop = prediction
    depth_scores, rotation, translation = score_scene(
        depth, poses, crop, batch.depth, batch.extrinsics,
    )
    metrics = {
        **pose_metrics(rotation, translation),
        "delta125": depth_scores["delta125"] * 100.0,
        "AbsRel": depth_scores["AbsRel"],
    }
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError(f"Non-finite metrics for frames {batch.frame_names}")
    return {
        "scored_frames": batch.frame_names,
        "pose_pairs": int(rotation.size),
        "metrics": metrics,
    }


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
    kernels = {}
    for i in sorted(selected_positions):
        filename = batch.frame_names[i]
        path = variants_dir / Path(filename).with_suffix(".png")
        kernels[filename] = write_blurred(batch.image_paths[i], path)
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
            "full": score_views(batch, prediction),
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
        }

    result = {
        "dataset": args.dataset, "scene": scene, "seed": args.seed,
        "checkpoint": args.checkpoint.name,
        "environment": {"torch": torch.__version__, "opencv": cv2.__version__,
                        "numpy": np.__version__, "device": args.device},
        "blur": {"kernel_xy": [next(iter(kernels.values())), 1] if len(set(kernels.values())) == 1 else None,
                 "source_kernel_widths": kernels, "reference_image_width": SINTEL_IMAGE_WIDTH,
                 "target_width_fraction": BLUR_WIDTH_FRACTION, "reference_kernel_width": KERNEL[0],
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
    return curve


def coverage_metrics(result):
    curve = {"10_0": result["runs"]["normal"]["full"]["metrics"]}
    for count in ORIGINAL_COUNTS:
        split = f"{10 - count}_{count}"
        row = result["splits"][split]
        curve[split] = row["mean_metrics"]
    return curve


def aggregate(curves):
    return {split: mean_metrics([curve[split] for curve in curves]) for split in curves[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sintel", "eth3d"), default="sintel")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPOSITORY / "checkpoint/vggt_omega_1b_416_reproduce.pt")
    parser.add_argument("--output-dir", type=Path, help="Directory cleared before each run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    frame_map = json.loads((ROOT / "frames" / f"{args.dataset}.json").read_text())
    args.scenes = args.scenes or list(frame_map)
    unknown = set(args.scenes).difference(frame_map)
    if unknown:
        parser.error(f"Unknown {args.dataset} scenes: {sorted(unknown)}")
    if args.output_dir is None:
        args.output_dir = REPOSITORY / "outputs/motion_blur" / RESULT_FOLDERS[args.dataset]
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
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    args.output_dir = output_dir
    per_scene = {
        scene: run_scene(args, scene, frame_map[scene], dataset, predictor)
        for scene in dict.fromkeys(args.scenes)
    }
    report = aggregate(list(per_scene.values()))
    report["per_scene"] = per_scene
    output = args.output_dir / "coverage_metrics.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
