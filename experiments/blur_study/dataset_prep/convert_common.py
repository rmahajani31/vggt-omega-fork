"""Shared CLI plumbing for the blur-study converters: scene selection, resume, workers, logs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPOSITORY = STUDY.parents[1]
for path in (REPOSITORY / "training", REPOSITORY, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.blur_schedule import read_manifest  # noqa: E402
from unified_io import SceneResult, config_hash, exr_roundtrip_probe, write_scene  # noqa: E402

MANIFEST = REPOSITORY / "data" / "vggt_omega_train_validation_v2.csv"
UNIFIED = REPOSITORY / "data" / "unified"


def manifest_rows(dataset_label: str, manifest: Path = MANIFEST, splits=("train", "validation")):
    rows = [r for r in read_manifest(manifest)
            if r["dataset"] == dataset_label and r["split"] in splits]
    return sorted(rows, key=lambda r: r["catalog_id"])


def base_parser(description: str, dataset_key: str, default_source: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", type=Path, default=default_source, help="raw dataset root")
    parser.add_argument("--dest", type=Path, default=UNIFIED / dataset_key, help="unified dataset root")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--scenes", nargs="+", help="catalog ids (default: all manifest rows)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace completed scenes (old scene kept until the new one validates)")
    return parser


def _convert_one(job):
    build, scene, args, version, cfg_hash, provenance = job
    try:
        frames, skipped = build(scene, args)
        result = write_scene(args["dest"], args["records"], scene, frames, version=version,
                             cfg_hash=cfg_hash, provenance=provenance, overwrite=args["overwrite"])
        result.skipped_frames = skipped
        return result
    except Exception as e:  # noqa: BLE001 - every failure is logged with its reason
        return SceneResult(scene, "failed", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def run(dataset_key: str, build, scenes, args, version: str, config: dict, provenance: dict):
    """Convert ``scenes`` with ``build(scene, args_dict) -> (frames, skipped)`` in parallel."""
    dest = Path(args.dest).resolve()
    records = dest.parent / "_records" / dataset_key
    logs = dest.parent / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    exr_roundtrip_probe(dest.parent / "_staging")
    cfg_hash = config_hash({"version": version, **config})
    arg_dict = {"dest": dest, "records": records, "overwrite": args.overwrite, **config}
    jobs = [(build, scene, arg_dict, version, cfg_hash, provenance) for scene in scenes]
    log_path = logs / f"{dataset_key}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    counts = {"converted": 0, "skipped": 0, "failed": 0}
    with open(log_path, "w", encoding="utf-8") as log:
        if args.workers <= 1:
            results = map(_convert_one, jobs)
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers)
            futures = [pool.submit(_convert_one, job) for job in jobs]
            results = (f.result() for f in as_completed(futures))
        for result in results:
            counts[result.status] += 1
            log.write(json.dumps({"scene": result.scene, "status": result.status, "frames": result.frames,
                                  "skipped_frames": result.skipped_frames,
                                  "warnings": result.validation.get("warnings", []),
                                  "error": result.error}, default=str) + "\n")
            log.flush()
            line = f"[{sum(counts.values())}/{len(jobs)}] {result.scene}: {result.status}"
            if result.status == "converted":
                line += f" ({result.frames} frames, {len(result.skipped_frames)} skipped)"
            if result.error:
                line += " -- " + result.error.splitlines()[0]
            print(line, flush=True)
    print(f"{dataset_key}: {counts} (log {log_path})")
    if counts["failed"]:
        raise SystemExit(1)
    return counts


def select(rows, args):
    scenes = [r["catalog_id"] for r in rows]
    if args.scenes:
        unknown = set(args.scenes) - set(scenes)
        if unknown:
            raise SystemExit(f"Not in the manifest: {sorted(unknown)}")
        scenes = [s for s in scenes if s in set(args.scenes)]
    return scenes


__all__ = ["base_parser", "manifest_rows", "run", "select"]
