"""
Central configuration for the leaf-disease explainability project.

Every path here is derived from the location of this file, so the whole
project can be downloaded, unzipped anywhere, and run without editing
a single path. Nothing below depends on an absolute path on my machine.

Layout expected after you place the dataset:

    leaf_xai/
        data/
            raw/<class_name>/<image>.jpg     <- PlantVillage colour images
            splits.csv                       <- generated once, shared by A1/A2
            classes.json                     <- generated once, shared by A1/A2
        outputs/
            checkpoints/  features/  figures/  metrics/
        src/
            config.py  data.py  utils.py  gradcam.py
            approach1_transfer.py  approach2_handcrafted.py
"""

from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root, i.e. the parent of src/)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # ImageFolder-style: raw/<class>/<img>.jpg
SPLIT_FILE = DATA_DIR / "splits.csv"
CLASSES_FILE = DATA_DIR / "classes.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
FEATURE_DIR = OUTPUT_DIR / "features"
FIG_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"

# --------------------------------------------------------------------------
# Reproducibility and split policy
# --------------------------------------------------------------------------
SEED = 42

# A1 and A2 MUST see exactly the same images in each split, otherwise the
# comparison between the two approaches is meaningless. That is why the split
# is computed once, written to splits.csv, and then re-read by both scripts.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --------------------------------------------------------------------------
# Image / model defaults (tuned for a Colab or Kaggle T4)
# --------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

# ImageNet statistics: required because A1 starts from ImageNet weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Two-stage fine-tuning schedule for Approach 1.
WARMUP_EPOCHS = 3        # classifier head only, backbone frozen
FINETUNE_EPOCHS = 12     # whole network unfrozen, small learning rate
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 4

# Approach 2 works on a per-class subsample by default: extracting GLCM
# texture over all ~54k PlantVillage images is slow and adds nothing to the
# conclusions. Set --max-per-class 0 to use everything.
A2_MAX_PER_CLASS = 400
A2_FEATURE_IMG_SIZE = 256   # images are resized to this before feature extraction


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in (DATA_DIR, OUTPUT_DIR, CKPT_DIR, FEATURE_DIR, FIG_DIR, METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_classes(classes: list[str]) -> None:
    """Persist the class list so the label -> index mapping never drifts."""
    ensure_dirs()
    CLASSES_FILE.write_text(json.dumps(list(classes), indent=2))


def load_classes() -> list[str]:
    """Read the frozen class list written by data.build_splits()."""
    if not CLASSES_FILE.exists():
        raise FileNotFoundError(
            f"{CLASSES_FILE} not found. Run:  python src/data.py --build"
        )
    return json.loads(CLASSES_FILE.read_text())
