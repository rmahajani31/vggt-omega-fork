"""Build the shared, immutable training schedule for every blur-study recipe.

Frame ids are drawn with each dataset's native UnifiedDataset sampler, instantiated from the
same blur_study config the trainer uses, so the frame distribution matches the authors' loader
settings. Run after prepare_splits.py and after the unified datasets exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from _paths import MANIFEST, STUDY_DIR, TRAINING

from data.blur_schedule import build_schedule, manifest_recordings, read_manifest
from data.dynamic_dataloader import DEFAULT_DISCRETE_ASPECT_RATIOS, _normalize_aspect_ratio_candidates


def aspect_ratio_candidates(common_config):
    if not getattr(common_config.augs, "use_discrete_aspect_ratios", False):
        raise SystemExit("The blur study expects use_discrete_aspect_ratios: true (authors' setting)")
    lo, hi = common_config.augs.aspects
    candidates = getattr(common_config.augs, "aspect_ratio_candidates", DEFAULT_DISCRETE_ASPECT_RATIOS)
    return [r for r in _normalize_aspect_ratio_candidates(candidates) if lo <= r <= hi]


def select_smoke_recordings(recordings, limit):
    """Deterministic, dataset-mixed subset for smoke schedules."""
    ranked = sorted(recordings, key=lambda r: hashlib.sha256(r["seq_name"].encode()).hexdigest())
    return sorted(ranked[:limit], key=lambda r: (r["dataset"], r["seq_name"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output", default=str(STUDY_DIR / "schedule.json"))
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=3,
                        help="sequences per step; 3 is the authors' FRAME_COUNT_TO_BATCH_SIZE[10]")
    parser.add_argument("--views", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--blur-px", type=float, default=31)
    parser.add_argument("--reference-width", type=float, default=846)
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke tests only: keep N training recordings")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*", help="hydra overrides for the blur_study config")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; the schedule is meant to be immutable (use --overwrite)")

    os.chdir(TRAINING)  # config paths are relative to training/, exactly as for launch.py
    from hydra.utils import instantiate
    from launch import load_config

    cfg = load_config("blur_study", args.overrides)
    common = cfg.data.train.common_config
    datasets = {key: instantiate(conf, common_conf=common)
                for key, conf in cfg.data.train.dataset.dataset_configs.items()}
    recordings = manifest_recordings(read_manifest(manifest), "train")
    if args.limit:
        recordings = select_smoke_recordings(recordings, args.limit)

    def sample_frame_ids(dataset_key, seq_name, num_views):
        return datasets[dataset_key].sample_frame_ids(seq_name, num_views)

    loader_settings = {
        key: {k: v for k, v in conf.items() if k != "_target_"}
        for key, conf in cfg.data.train.dataset.dataset_configs.items()
    }
    schedule = build_schedule(
        recordings, sample_frame_ids, aspect_ratio_candidates(common),
        passes=args.passes, batch_size=args.batch_size, num_views=args.views, seed=args.seed,
        blur_px=args.blur_px, reference_width=args.reference_width,
        metadata={
            "manifest": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "img_size": int(common.img_size),
            "smoke_limit": args.limit,
            "loader_settings": loader_settings,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schedule, indent=1, default=str) + "\n", encoding="utf-8")
    n = len(schedule["schedule"][0]["batches"])
    print(f"Wrote {output}: {args.passes} passes x {n} batches of {args.batch_size}")


if __name__ == "__main__":
    main()
