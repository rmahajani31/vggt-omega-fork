"""Minimal LoRA adapters for the blur-study fine-tuning runs.

``apply_lora`` wraps selected ``nn.Linear`` modules (including subclasses such as
``LinearKMaskedBias``) in place and freezes every original parameter. ``merge_lora`` returns a
plain state dict with the original key names, so merged checkpoints load into ``VGGTOmega()``
with ``strict=True`` exactly like the released checkpoints.
"""

from __future__ import annotations

import fnmatch
import logging
import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

LORA_KEYS = ("lora_A", "lora_B")


class LoRALinear(nn.Module):
    """``base(x) + (alpha / r) * B(A(dropout(x)))`` with B initialised to zero."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device,
                                               dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device,
                                               dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        update = nn.functional.linear(nn.functional.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B.float() @ self.lora_A.float()) * self.scaling


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def apply_lora(model: nn.Module, target_patterns: List[str], r: int = 16, alpha: float = 32.0,
               dropout: float = 0.0, exclude_patterns: List[str] | None = None) -> List[str]:
    """Freeze ``model`` and wrap every ``nn.Linear`` whose module name matches a pattern.

    Patterns use fnmatch syntax over full module names, e.g. ``aggregator.frame_blocks.*``.
    Every pattern must match at least one linear layer. Returns the wrapped module names.
    """
    exclude_patterns = list(exclude_patterns or [])
    for param in model.parameters():
        param.requires_grad_(False)

    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, target_patterns)
        and not _matches(name, exclude_patterns)
    ]
    for pattern in target_patterns:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in targets):
            raise ValueError(f"LoRA pattern '{pattern}' matched no nn.Linear module")

    for name in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), r, alpha, dropout))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"LoRA: wrapped {len(targets)} linear layers (r={r}, alpha={alpha}); "
                 f"{trainable:,} trainable of {total:,} parameters")
    return targets


def _lora_modules(model: nn.Module):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Adapter weights only, keyed by their full parameter names."""
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if name.rsplit(".", 1)[-1] in LORA_KEYS}


@torch.no_grad()
def merge_lora(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state dict with adapters folded in and original (pre-LoRA) key names.

    The model itself is left unchanged.
    """
    deltas = {f"{name}.base.weight": module.delta_weight() for name, module in _lora_modules(model)}
    merged = {}
    for key, value in model.state_dict().items():
        if key.rsplit(".", 1)[-1] in LORA_KEYS:
            continue
        if key in deltas:
            value = (value.float() + deltas[key].to(value.device)).to(value.dtype)
        merged[key.replace(".base.", ".")] = value.detach().cpu()
    return merged


@torch.no_grad()
def merge_lora_state_dict(state: Dict[str, torch.Tensor], scaling: float) -> Dict[str, torch.Tensor]:
    """Merge a LoRA-wrapped model's state dict (``.base.`` / ``.lora_A`` keys) without a model."""
    merged = {}
    prefixes = {key[: -len(".lora_A")] for key in state if key.endswith(".lora_A")}
    for key, value in state.items():
        if key.rsplit(".", 1)[-1] in LORA_KEYS:
            continue
        head, sep, tail = key.rpartition(".base.")
        if sep and head in prefixes:
            if tail == "weight":
                delta = (state[f"{head}.lora_B"].float() @ state[f"{head}.lora_A"].float()) * scaling
                value = (value.float() + delta.to(value.device)).to(value.dtype)
            key = f"{head}.{tail}"
        merged[key] = value.detach().cpu()
    return merged


def merged_from_lora_checkpoint(base_state: Dict[str, torch.Tensor],
                                lora_state: Dict[str, torch.Tensor],
                                scaling: float) -> Dict[str, torch.Tensor]:
    """Merge a saved adapter dict into a base (original-key) state dict without a model."""
    merged = dict(base_state)
    prefixes = {key.rsplit(".", 1)[0] for key in lora_state}
    for prefix in prefixes:
        weight_key = f"{prefix}.weight"
        if weight_key not in merged:
            raise KeyError(f"Base state has no {weight_key} for adapter {prefix}")
        delta = (lora_state[f"{prefix}.lora_B"].float() @ lora_state[f"{prefix}.lora_A"].float()) * scaling
        merged[weight_key] = (merged[weight_key].float() + delta).to(merged[weight_key].dtype)
    return merged
