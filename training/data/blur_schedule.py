from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
