from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


# Manifest dataset label -> training config key.
MANIFEST_DATASETS = {"uCO3D": "uco3d", "Hypersim": "hypersim", "MVS-Synth": "mvssynth"}


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
