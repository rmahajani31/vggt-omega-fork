"""Repository paths for the blur-study scripts; importing this makes training/ importable."""

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY = STUDY_DIR.parents[1]
TRAINING = REPOSITORY / "training"
EVAL = REPOSITORY / "eval"
DATA = REPOSITORY / "data"
MANIFEST = DATA / "vggt_omega_train_validation_v2.csv"
UNIFIED = DATA / "unified"
SPLITS = STUDY_DIR / "splits"

for path in (TRAINING, REPOSITORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
