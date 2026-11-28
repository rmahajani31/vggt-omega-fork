"""Add the randomized-severity recipes width2to10_a and width2to10_b to the shared schedule.

Both recipes reuse every sample of experiments/blur_study/schedule.json (recordings, frames,
aspect ratios, augmentation seeds and blur permutations) and give each blurred view its own
horizontal box-blur width as a fraction of that view's source image width, drawn uniformly
from [0.02, 0.10]. The fraction is stored per sample in ``width_fraction_by_view[recipe]``
(10 entries, 0 for sharp views) and applied at source resolution by
``data/datasets/unified.py`` before any crop or resize.

  width2to10_a  blurs exactly the views `variable` blurs (150 each of 1/2/3 blurred views per
                pass), so A minus variable isolates the change of severity distribution.
  width2to10_b  per pass, moves 45 of A's one-blur recordings to fully sharp, and adds one view
                to 15 other one-blur and 30 two-blur recordings (45/90/135/180 recordings with
                0/1/2/3 blurred views). Views blurred by both recipes keep A's width; the 45
                widths freed by the sharp recordings go to the 45 newly blurred views. The
                900 widths per pass are therefore identical, so B minus A isolates the blur-
                count distribution.

Output is deterministic. The run in outputs/randomized-blur-width2to10/20261127T185344Z used
the file this script writes (sha256 checked with --check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from _paths import STUDY_DIR

from data.blur_schedule import (WIDTH_FRACTION_RANGE, WIDTH_RECIPES, blurred_positions,
                                stable_seed, validate_schedule)

# sha256 of the pilot schedule.json with LF line endings; the width draws depend on its order.
PILOT_SCHEDULE_SHA256 = "dc885911ae65251f27d59aba3c3fe8647d7a2c0a87bd183af14e094e9fa49eff"
# sha256 of the schedule this script produced for the 20261127T185344Z training run.
RUN_SCHEDULE_SHA256 = "112280f4e37a0a777a5bfee231847b8c5b1ea2227987e349e668327d40da02a5"


def add_width_recipes(schedule: dict) -> list[dict]:
    """Add both recipes to ``schedule`` in place; return a per-pass audit of the width draws."""
    A, B = WIDTH_RECIPES
    low, high = WIDTH_FRACTION_RANGE
    validate_schedule(schedule)
    assert schedule["num_views"] == 10 and len(schedule["schedule"]) == 5
    audit = []
    for epoch in schedule["schedule"]:
        samples = [s for b in epoch["batches"] for s in b["samples"]]
        assert len(epoch["batches"]) == 150 and len(samples) == 450
        rng = random.Random(stable_seed(schedule["seed"], "severity-v1", epoch["pass"]))
        groups = {k: [i for i, s in enumerate(samples) if s["blur_count"]["variable"] == k]
                  for k in (1, 2, 3)}
        for indices in groups.values():
            assert len(indices) == 150
            rng.shuffle(indices)
        for s in samples:
            s["blur_count"][A] = s["blur_count"][B] = s["blur_count"]["variable"]
        # Change the fewest samples while keeping B's blurred views a prefix of each blur_perm:
        # 45 transitions 1->0, 15 transitions 1->2, 30 transitions 2->3.
        for ids, count in ((groups[1][:45], 0), (groups[1][45:60], 2), (groups[2][:30], 3)):
            for i in ids:
                samples[i]["blur_count"][B] = count
        slots = {r: [(i, p) for i, s in enumerate(samples) for p in blurred_positions(s, r)]
                 for r in (A, B)}
        assert len(slots[A]) == len(slots[B]) == 900
        width_rng = random.Random(stable_seed(schedule["seed"], "uniform-width-2-to-10", epoch["pass"]))
        pool = [width_rng.uniform(low, high) for _ in slots[A]]
        assigned_a = dict(zip(slots[A], pool))
        lost = sorted(set(slots[A]) - set(slots[B]))
        gained = sorted(set(slots[B]) - set(slots[A]))
        assert len(lost) == len(gained) == 45
        recycled = [assigned_a[slot] for slot in lost]
        rng.shuffle(recycled)
        assigned_b = {slot: assigned_a[slot] for slot in slots[B] if slot in assigned_a}
        assigned_b.update(zip(gained, recycled))
        for s in samples:
            s["width_fraction_by_view"] = {}
        for r, assignments in ((A, assigned_a), (B, assigned_b)):
            for i, s in enumerate(samples):
                v = [assignments.get((i, pos), 0) for pos in range(10)]
                assert v[0] == 0 and [p for p, w in enumerate(v) if w] == blurred_positions(s, r)
                s["width_fraction_by_view"][r] = v
                if r == A:
                    assert blurred_positions(s, A) == blurred_positions(s, "variable")
            counts = Counter(s["blur_count"][r] for s in samples)
            fractions = sorted(assignments.values())
            assert counts == ({1: 150, 2: 150, 3: 150} if r == A else {0: 45, 1: 90, 2: 135, 3: 180})
            assert fractions == sorted(pool) and low <= fractions[0] <= fractions[-1] <= high
            bands = [0] * 4
            for fraction in fractions:
                bands[sum(fraction >= boundary for boundary in (0.04, 0.06, 0.08))] += 1
            audit.append(dict(epoch=epoch["pass"], recipe=r, counts=counts, selected_views=len(fractions),
                              min_fraction=fractions[0], mean_fraction=math.fsum(fractions) / len(fractions),
                              max_fraction=fractions[-1], counts_in_2_4_6_8_10_percent_bins=bands))
    schedule["recipes"].extend((A, B))
    validate_schedule(schedule)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, default=STUDY_DIR / "schedule.json")
    parser.add_argument("--output", type=Path, default=STUDY_DIR / "schedule_width2to10.json")
    parser.add_argument("--audit", type=Path, default=STUDY_DIR / "schedule_width2to10_audit.json")
    parser.add_argument("--check", action="store_true",
                        help="do not write; verify that --output matches a fresh build and the training run")
    args = parser.parse_args()

    raw = args.pilot.read_bytes()
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != PILOT_SCHEDULE_SHA256:
        raise SystemExit(f"{args.pilot} is not the pilot schedule this recipe was built from")
    schedule = json.loads(raw)
    audit = add_width_recipes(schedule)
    text = json.dumps(schedule, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if args.check:
        on_disk = hashlib.sha256(args.output.read_bytes()).hexdigest()
        if not on_disk == digest == RUN_SCHEDULE_SHA256:
            raise SystemExit(f"Mismatch: built {digest}, {args.output} {on_disk}, run {RUN_SCHEDULE_SHA256}")
        print(f"{args.output} matches a fresh build and the training run ({digest[:12]})")
        return
    args.output.write_text(text, encoding="utf-8")
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} (sha256 {digest[:12]}) and {args.audit}: "
          "five passes with 900 matched width fractions per recipe and pass.")


if __name__ == "__main__":
    main()
