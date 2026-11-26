from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPOSITORY = STUDY.parents[1]
for path in (REPOSITORY / "training", REPOSITORY, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.blur_schedule import read_manifest  # noqa: E402

MANIFEST = REPOSITORY / "data" / "vggt_omega_train_validation_v2.csv"
UNIFIED = REPOSITORY / "data" / "unified"


def manifest_rows(dataset_label: str, manifest: Path = MANIFEST, splits=("train", "validation")):
    rows = [r for r in read_manifest(manifest)
            if r["dataset"] == dataset_label and r["split"] in splits]
    return sorted(rows, key=lambda r: r["catalog_id"])
