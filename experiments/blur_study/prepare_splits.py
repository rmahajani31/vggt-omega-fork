"""Write per-dataset train/val sequence lists from the study manifest and check them.

Checks: exact cohort sizes (225/125/100 train, 25/15/10 source-val), no recording in both
splits, every id present in the authors' training/valid_seqs lists, and (with --check-data)
every scene present in the unified dataset root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _paths import MANIFEST, SPLITS, TRAINING, UNIFIED
from data.blur_schedule import MANIFEST_DATASETS, manifest_recordings, read_manifest

EXPECTED = {
    "train": {"uco3d": 225, "hypersim": 125, "mvssynth": 100},
    "validation": {"uco3d": 25, "hypersim": 15, "mvssynth": 10},
}
VALID_SEQS = {"uco3d": "uco3d.txt", "hypersim": "hypersim.txt", "mvssynth": "mvssynth.txt"}


def load_splits(manifest):
    rows = read_manifest(manifest)
    return {
        "train": manifest_recordings(rows, "train"),
        "validation": manifest_recordings(rows, "validation", cohort="source_val50"),
    }


def check_splits(splits, valid_seqs_dir):
    for split, recs in splits.items():
        counts = dict(Counter(r["dataset"] for r in recs))
        if counts != EXPECTED[split]:
            raise SystemExit(f"{split} counts {counts} != expected {EXPECTED[split]}")
    overlap = ({(r["dataset"], r["seq_name"]) for r in splits["train"]}
               & {(r["dataset"], r["seq_name"]) for r in splits["validation"]})
    if overlap:
        raise SystemExit(f"{len(overlap)} recordings are in both splits: {sorted(overlap)[:5]}")
    for key, filename in VALID_SEQS.items():
        allowed = set((Path(valid_seqs_dir) / filename).read_text().split())
        for split, recs in splits.items():
            unknown = [r["seq_name"] for r in recs if r["dataset"] == key and r["seq_name"] not in allowed]
            if unknown:
                raise SystemExit(f"{len(unknown)} {key} {split} ids are not in valid_seqs: {unknown[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--out", default=SPLITS)
    parser.add_argument("--unified-root", default=UNIFIED)
    parser.add_argument("--check-data", action="store_true",
                        help="also require every scene to exist in the unified root")
    args = parser.parse_args()

    splits = load_splits(args.manifest)
    check_splits(splits, TRAINING / "valid_seqs")

    if args.check_data:
        missing = [f"{r['dataset']}/{r['seq_name']}" for recs in splits.values() for r in recs
                   if not (Path(args.unified_root) / r["dataset"] / r["seq_name"] / "image_names.json").is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} scenes missing from {args.unified_root}: {missing[:10]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = {"train": "train", "validation": "val"}
    for split, recs in splits.items():
        for key in MANIFEST_DATASETS.values():
            names = [r["seq_name"] for r in recs if r["dataset"] == key]
            (out / f"{key}_{tag[split]}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote split lists to {out}: " + ", ".join(f"{s} {len(r)}" for s, r in splits.items()))


if __name__ == "__main__":
    main()
