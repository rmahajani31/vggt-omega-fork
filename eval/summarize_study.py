"""Summarize the blur study into one results table per validation cohort.

Reads outputs/blur_study/<cohort>/<model>/coverage_metrics.json written by motion_blur.py and
writes <out>/<cohort>.md and <cohort>.csv. Each row is one model and one blur count (0-5); the
six scores are AUC@3 and AUC@30 for the full, clean-only and blurred-only pair groups on a
0-100 scale. Blur-trained recipes also show the signed difference from clean fine-tuning in
percentage points for the same cohort, blur count, pair group and threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent

COHORTS = {"source_val": "Held-out training-source recordings (50)",
           "sintel": "Sintel (16 sequences)", "eth3d": "ETH3D (13 scenes)"}
MODELS = {
    "original": "Original checkpoint",
    "clean": "Clean fine-tuning",
    "fixed": "Fixed fraction (2)",
    "variable": "Variable fraction (1/2/3)",
    "clean_or_heavy": "Clean and heavy blur (0/4)",
}
BASELINE = "clean"
BLUR_TRAINED = ("fixed", "variable", "clean_or_heavy")
GROUPS = ("full", "clean_only", "blurred_only")
THRESHOLDS = ("AUC@3", "AUC@30")
BLUR_COUNTS = range(0, 6)


def split_name(count: int, views: int = 10) -> str:
    return f"{views - count}_{count}"


def load_curves(path: Path) -> dict[str, dict[str, dict]]:
    """Return {group: {split: metrics}} from a coverage_metrics.json report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    curves = {"full": {k: v for k, v in report.items() if "_" in k and k[0].isdigit()}}
    for group in ("clean_only", "blurred_only"):
        curves[group] = {k: v for k, v in report[group].items() if "_" in k and k[0].isdigit()}
    return curves


def score(curves, group, count, threshold):
    metrics = curves[group].get(split_name(count))
    value = None if metrics is None else metrics.get(threshold)
    return None if value is None else float(value)


def check_curves(model, curves):
    for threshold in THRESHOLDS:
        full, clean = score(curves, "full", 0, threshold), score(curves, "clean_only", 0, threshold)
        if full is None or clean is None or not math.isclose(full, clean, abs_tol=1e-9):
            raise ValueError(f"{model}: full and clean-only {threshold} differ at 0 blurred views")
        for count in (0, 1):
            if score(curves, "blurred_only", count, threshold) is not None:
                raise ValueError(f"{model}: blurred-only {threshold} must be N/A at {count} views")


def build_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    baseline = results.get(BASELINE)
    for model in MODELS:
        if model not in results:
            continue
        for count in BLUR_COUNTS:
            row = {"model": model, "label": MODELS[model], "blurred_views": count}
            for group in GROUPS:
                for threshold in THRESHOLDS:
                    key = f"{group}:{threshold}"
                    value = score(results[model], group, count, threshold)
                    row[key] = value
                    delta = None
                    if model in BLUR_TRAINED and baseline is not None and value is not None:
                        base = score(baseline, group, count, threshold)
                        delta = None if base is None else value - base
                    row[f"{key}:delta"] = delta
            rows.append(row)
    return rows


def fmt(value, delta=None, show_delta=False):
    if value is None:
        return "N/A"
    text = f"{value:.1f}"
    if show_delta and delta is not None:
        text += f" ({delta:+.1f})"
    return text


def write_markdown(path: Path, cohort: str, rows: list[dict], scenes: int | None = None) -> None:
    header = ["Model", "Blurred views"] + [
        f"{g.replace('_', '-')} {t}" for g in GROUPS for t in THRESHOLDS]
    title = COHORTS.get(cohort, cohort)
    if scenes is not None:
        title += f" - {scenes} scenes evaluated"
    lines = [f"# {title}", "",
             "AUC on a 0-100 scale; parentheses give the signed difference from clean "
             "fine-tuning in percentage points. N/A: no pairs in that group.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["label"], str(row["blurred_views"])]
        for g in GROUPS:
            for t in THRESHOLDS:
                key = f"{g}:{t}"
                cells.append(fmt(row[key], row[f"{key}:delta"], row["model"] in BLUR_TRAINED))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["model", "label", "blurred_views"] + [
        f"{g}:{t}{suffix}" for g in GROUPS for t in THRESHOLDS for suffix in ("", ":delta")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPOSITORY / "outputs/blur_study")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    args = parser.parse_args()
    out = args.out or args.results_root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    for cohort in args.cohorts:
        results, scene_sets = {}, {}
        for model in MODELS:
            path = args.results_root / cohort / model / "coverage_metrics.json"
            if path.is_file():
                results[model] = load_curves(path)
                check_curves(model, results[model])
                meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
                scene_sets[model] = tuple(sorted(meta.get("scenes", [])))
        if len(set(scene_sets.values())) > 1:
            raise ValueError(f"{cohort}: models were evaluated on different scenes; deltas would be invalid")
        if not results:
            print(f"{cohort}: no results under {args.results_root / cohort}")
            continue
        missing = [m for m in MODELS if m not in results]
        rows = build_rows(results)
        scenes = len(next(iter(scene_sets.values()))) if scene_sets else None
        write_markdown(out / f"{cohort}.md", cohort, rows, scenes)
        write_csv(out / f"{cohort}.csv", rows)
        print(f"{cohort}: {len(results)} models -> {out / (cohort + '.md')}"
              + (f" (missing: {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
