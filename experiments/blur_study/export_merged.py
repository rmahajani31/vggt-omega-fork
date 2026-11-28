"""Export an evaluation checkpoint (original key names, adapters merged) for each recipe.

The trainer already writes logs/blur_study_<recipe>/checkpoints/model_merged.pt at the end of
training. This script rebuilds it from the last training checkpoint (e.g. after an interrupted
run that was resumed, or to merge an intermediate checkpoint_<epoch>.pt), then verifies that
it loads into VGGTOmega() with strict=True.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _paths import REPOSITORY, TRAINING
from train_utils.lora import merge_lora_state_dict
from vggt_omega.models import VGGTOmega

from data.blur_schedule import RECIPES  # noqa: E402  (training/ is importable via _paths)


def export(checkpoint: Path, output: Path, r: int, alpha: float) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    merged = merge_lora_state_dict(state, alpha / r)
    VGGTOmega().load_state_dict(merged, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(f"{checkpoint} -> {output} (strict load OK)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", nargs="+", default=list(RECIPES))
    parser.add_argument("--logs", type=Path, default=TRAINING / "logs")
    parser.add_argument("--checkpoint-name", default="checkpoint.pt")
    parser.add_argument("--out", type=Path, default=REPOSITORY / "checkpoint/blur_study")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--checkpoint", type=Path, help="export one checkpoint file instead of --recipes")
    parser.add_argument("--output", type=Path, help="output path for --checkpoint")
    args = parser.parse_args()
    if args.checkpoint is not None:
        if args.output is None:
            parser.error("--checkpoint requires --output")
        export(args.checkpoint, args.output, args.r, args.alpha)
        return
    for recipe in args.recipes:
        source = args.logs / f"blur_study_{recipe}" / "checkpoints" / args.checkpoint_name
        export(source, args.out / f"{recipe}.pt", args.r, args.alpha)


if __name__ == "__main__":
    main()
