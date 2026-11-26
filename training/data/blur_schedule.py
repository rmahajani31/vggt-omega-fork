"""Matched-sample schedule for the motion-blur fine-tuning study.

One immutable schedule is shared by every recipe. For each pass over the training recordings
it fixes the batch composition, each batch's aspect ratio, every sample's frame indices, the
seed that drives crop/scale/colour augmentation, and a permutation of the blur-eligible view
positions. A recipe only chooses how many views of that permutation to blur, so all recipes
see identical recordings, frames, view order and augmentation; only the blurred pixels differ.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

SCHEDULE_VERSION = 1

# Manifest dataset label -> training config key.
MANIFEST_DATASETS = {"uCO3D": "uco3d", "Hypersim": "hypersim", "MVS-Synth": "mvssynth"}

RECIPES = ("clean", "fixed", "variable", "clean_or_heavy")


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_recordings(rows: Iterable[Mapping[str, str]], split: str,
                        cohort: str | None = None) -> List[Dict[str, str]]:
    """Return ``{"dataset": key, "seq_name": catalog_id}`` for one split, sorted."""
    out = []
    for row in rows:
        if row["split"] != split or (cohort is not None and row["cohort"] != cohort):
            continue
        if row["dataset"] not in MANIFEST_DATASETS:
            continue
        out.append({"dataset": MANIFEST_DATASETS[row["dataset"]], "seq_name": row["catalog_id"]})
    out.sort(key=lambda r: (r["dataset"], r["seq_name"]))
    keys = [(r["dataset"], r["seq_name"]) for r in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate recordings in manifest split {split}")
    return out


def balanced_counts(n: int, values: Sequence[int], rng: random.Random) -> List[int]:
    """Exactly balanced assignment of ``values`` over ``n`` samples, shuffled."""
    if n % len(values) != 0:
        raise ValueError(f"{n} samples cannot be split evenly over {list(values)}")
    counts = [v for v in values for _ in range(n // len(values))]
    rng.shuffle(counts)
    return counts


def recipe_counts(recipe: str, n: int, rng: random.Random) -> List[int]:
    if recipe == "clean":
        return [0] * n
    if recipe == "fixed":
        return [2] * n
    if recipe == "variable":
        return balanced_counts(n, (1, 2, 3), rng)
    if recipe == "clean_or_heavy":
        return balanced_counts(n, (0, 4), rng)
    raise ValueError(f"Unknown recipe {recipe}")


def build_schedule(
    recordings: Sequence[Mapping[str, str]],
    sample_frame_ids: Callable[[str, str, int], Sequence[int]],
    aspect_ratio_candidates: Sequence[float],
    *,
    passes: int = 5,
    batch_size: int = 3,
    num_views: int = 10,
    seed: int = 20261125,
    blur_px: float = 31,
    reference_width: float = 846,
    metadata: Mapping | None = None,
) -> Dict:
    """Build the shared schedule.

    ``sample_frame_ids(dataset_key, seq_name, num_views)`` must draw frames with the dataset's
    native sampler using the global ``random``/``np.random`` state; it is called right after
    both are seeded with a per-sample seed, so the draw is reproducible.
    """
    if len(recordings) % batch_size != 0:
        raise ValueError(f"{len(recordings)} recordings do not form whole batches of {batch_size}")
    eligible = list(range(1, num_views))
    schedule = {
        "version": SCHEDULE_VERSION,
        "seed": seed,
        "passes": passes,
        "batch_size": batch_size,
        "num_views": num_views,
        "blur": {"blur_px": blur_px, "reference_width": reference_width},
        "recipes": list(RECIPES),
        "metadata": dict(metadata or {}),
        "schedule": [],
    }
    for pass_idx in range(passes):
        rng = random.Random(stable_seed(seed, "pass", pass_idx))
        order = list(recordings)
        rng.shuffle(order)
        counts = {recipe: recipe_counts(recipe, len(order), rng) for recipe in RECIPES}
        samples = []
        for i, rec in enumerate(order):
            sample_seed = stable_seed(seed, "frames", pass_idx, rec["dataset"], rec["seq_name"])
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            frame_ids = [int(x) for x in sample_frame_ids(rec["dataset"], rec["seq_name"], num_views)]
            if len(frame_ids) != num_views:
                raise ValueError(f"Sampler returned {len(frame_ids)} ids for {rec['seq_name']}")
            perm_rng = random.Random(stable_seed(seed, "blur_perm", pass_idx, rec["seq_name"]))
            samples.append({
                "dataset": rec["dataset"],
                "seq_name": rec["seq_name"],
                "frame_ids": frame_ids,
                "aug_seed": stable_seed(seed, "aug", pass_idx, rec["dataset"], rec["seq_name"]),
                "blur_perm": perm_rng.sample(eligible, len(eligible)),
                "blur_count": {recipe: counts[recipe][i] for recipe in RECIPES},
            })
        batches = []
        for b in range(0, len(samples), batch_size):
            batches.append({
                "aspect_ratio": float(rng.choice(list(aspect_ratio_candidates))),
                "samples": samples[b:b + batch_size],
            })
        schedule["schedule"].append({"pass": pass_idx, "batches": batches})
    validate_schedule(schedule, {(r["dataset"], r["seq_name"]) for r in recordings})
    return schedule


def blurred_positions(sample: Mapping, recipe: str) -> List[int]:
    return sorted(sample["blur_perm"][: sample["blur_count"][recipe]])


def validate_schedule(schedule: Mapping, expected_recordings: set | None = None) -> None:
    """Check the invariants the study depends on; raise ValueError on any violation."""
    if schedule.get("version") != SCHEDULE_VERSION:
        raise ValueError(f"Unsupported schedule version {schedule.get('version')}")
    num_views = schedule["num_views"]
    for entry in schedule["schedule"]:
        samples = [s for batch in entry["batches"] for s in batch["samples"]]
        if any(len(batch["samples"]) != schedule["batch_size"] for batch in entry["batches"]):
            raise ValueError(f"Pass {entry['pass']} has a partial batch")
        keys = [(s["dataset"], s["seq_name"]) for s in samples]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Pass {entry['pass']} visits a recording twice")
        if expected_recordings is not None and set(keys) != set(expected_recordings):
            raise ValueError(f"Pass {entry['pass']} does not cover exactly the training recordings")
        for s in samples:
            if sorted(s["blur_perm"]) != list(range(1, num_views)):
                raise ValueError(f"Bad blur permutation for {s['seq_name']} (view 0 must stay clean)")
            if len(s["frame_ids"]) != num_views:
                raise ValueError(f"Bad frame list for {s['seq_name']}")
        n = len(samples)
        totals = {r: Counter(s["blur_count"][r] for s in samples) for r in schedule["recipes"]}
        expected = {
            "clean": Counter({0: n}),
            "fixed": Counter({2: n}),
            "variable": Counter({1: n // 3, 2: n // 3, 3: n // 3}),
            "clean_or_heavy": Counter({0: n // 2, 4: n // 2}),
        }
        for recipe in schedule["recipes"]:
            if totals[recipe] != expected[recipe]:
                raise ValueError(f"Pass {entry['pass']} {recipe} counts {dict(totals[recipe])} "
                                 f"!= {dict(expected[recipe])}")


def load_schedule(path: str | Path) -> Dict:
    schedule = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_schedule(schedule)
    return schedule


def schedule_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
