"""Schedule-driven training data for the motion-blur study.

``ScheduledTorchDataset`` replaces ``DynamicTorchDataset``: instead of drawing sequences, frame
counts and aspect ratios at random, it replays one pass of a precomputed schedule per epoch
(see ``data/blur_schedule.py``). ``ScheduledBlurDataset`` loads each scheduled sample through
the native ``UnifiedDataset`` path with fixed frame ids, blurs the recipe's views at source
resolution, and reuses ``ComposedDataset.postprocess`` for augmentation, tracks and
normalisation.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from .blur_schedule import (RANDOM_WIDTH_RECIPES, RECIPES, blurred_positions, load_schedule,
                            schedule_sha256)
from .composed_dataset import ComposedDataset
from .worker_fn import get_worker_init_fn


class ScheduledBlurDataset(ComposedDataset):
    def __init__(self, dataset_configs, common_config, schedule_path, recipe):
        super().__init__(dataset_configs, common_config)
        if recipe not in RECIPES:
            raise ValueError(f"recipe must be one of {RECIPES}, got {recipe!r}")
        self.recipe = recipe
        self.schedule_path = str(schedule_path)
        self.schedule = load_schedule(schedule_path)
        if recipe not in self.schedule["recipes"]:
            raise ValueError(f"{self.schedule_path} has no {recipe!r} recipe; rebuild it with "
                             "experiments/blur_study/build_schedule.py")
        self.blur_cfg = dict(self.schedule["blur"])
        self.num_views = self.schedule["num_views"]
        missing = {s["dataset"] for p in self.schedule["schedule"] for b in p["batches"]
                   for s in b["samples"]} - set(self.datasets_by_key)
        if missing:
            raise ValueError(f"Schedule uses datasets missing from dataset_configs: {sorted(missing)}")
        logging.info(f"Blur schedule {self.schedule_path} (sha256 {schedule_sha256(schedule_path)[:12]}), "
                     f"recipe={recipe}, {len(self.schedule['schedule'])} passes")

    def __len__(self):
        return sum(len(b["samples"]) for p in self.schedule["schedule"] for b in p["batches"])

    def sample_entry(self, pass_idx, batch_idx, slot):
        batch = self.schedule["schedule"][pass_idx]["batches"][batch_idx]
        return batch, batch["samples"][slot]

    def __getitem__(self, index):
        pass_idx, batch_idx, slot = index
        batch_entry, sample = self.sample_entry(pass_idx, batch_idx, slot)
        # Everything stochastic downstream (scale crop, principal-point jitter, colour jitter,
        # JPEG, tracks) is driven by this seed, identically for every recipe and worker.
        seed = int(sample["aug_seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dataset = self.datasets_by_key[sample["dataset"]]
        raw = dataset.get_data(
            seq_name=sample["seq_name"],
            img_per_seq=self.num_views,
            ids=sample["frame_ids"],
            aspect_ratio=batch_entry["aspect_ratio"],
            exact_ids=True,
            blur_views=blurred_positions(sample, self.recipe),
            # Random-width recipes blur each selected view with its own scheduled width fraction.
            blur_cfg=({"width_fractions": sample["width_fraction_by_view"][self.recipe]}
                      if self.recipe in RANDOM_WIDTH_RECIPES else self.blur_cfg),
        )
        out = self.postprocess(raw)
        out["blur_mask"] = torch.zeros(self.num_views, dtype=torch.bool)
        out["blur_mask"][blurred_positions(sample, self.recipe)] = True
        return out


class ScheduledTorchDataset:
    """Drop-in for DynamicTorchDataset: epoch ``e`` replays schedule pass ``e``."""

    def __init__(self, dataset, common_config, num_workers, pin_memory=True, seed=0,
                 use_spawn=False, worker_init_fn=None, collate_fn=None, **_unused):
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.use_spawn = use_spawn and num_workers > 0
        self.worker_init_fn = worker_init_fn
        self.collate_fn = collate_fn
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

    def batches_for_epoch(self, epoch, rank=0, world_size=1):
        passes = self.dataset.schedule["schedule"]
        if epoch >= len(passes):
            raise ValueError(f"Schedule has {len(passes)} passes; epoch {epoch} is out of range")
        batches = [[(epoch, b, j) for j in range(len(batch["samples"]))]
                   for b, batch in enumerate(passes[epoch]["batches"])]
        return batches[rank::world_size]

    def get_loader(self, epoch):
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        batches = self.batches_for_epoch(int(epoch), rank, world_size)
        logging.info(f"Scheduled loader: pass {epoch}, {len(batches)} batches on rank {rank}")
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "batch_sampler": batches,
            "collate_fn": self.collate_fn,
            "worker_init_fn": get_worker_init_fn(seed=self.seed, epoch=epoch,
                                                 worker_init_fn=self.worker_init_fn),
        }
        if self.use_spawn:
            kwargs["multiprocessing_context"] = "spawn"
        return DataLoader(self.dataset, **kwargs)
