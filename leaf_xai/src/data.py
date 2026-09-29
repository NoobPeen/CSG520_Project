"""
Dataset discovery and the single train/val/test split shared by all approaches.

This module deliberately depends only on pandas / scikit-learn (no torch), so
Approach 2 can run on a machine without a deep-learning stack installed.
The torch-specific Dataset and DataLoader code lives in torch_data.py.

Run once, before anything else:

    python src/data.py --build

That writes data/splits.csv and data/classes.json. Both A1 and A2 read those
files, which is what guarantees the two approaches are evaluated on exactly
the same held-out images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Classes with fewer than this many images cannot be split three ways in a
# stratified manner, and are too small to learn from anyway.
MIN_IMAGES_PER_CLASS = 12


# --------------------------------------------------------------------------
# 1. Discovery
# --------------------------------------------------------------------------
def discover_images(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """
    Walk raw_dir/<class_name>/<image> and return a DataFrame of
    (relpath, label). relpath is relative to raw_dir and stored with forward
    slashes so the CSV works identically on Windows, Linux and Colab.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory {raw_dir} does not exist.\n"
            "Place the PlantVillage colour images as data/raw/<class>/<img>.jpg"
        )

    records = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.suffix.lower() in IMAGE_EXTS:
                records.append(
                    {
                        "relpath": img_path.relative_to(raw_dir).as_posix(),
                        "label": class_dir.name,
                    }
                )

    if not records:
        raise RuntimeError(
            f"No images found under {raw_dir}. Expected data/raw/<class>/<img>.jpg"
        )

    df = pd.DataFrame.from_records(records)

    # Drop classes that are too small to split or to train on.
    counts = df["label"].value_counts()
    too_small = counts[counts < MIN_IMAGES_PER_CLASS].index.tolist()
    if too_small:
        print(f"[data] dropping {len(too_small)} class(es) with "
              f"< {MIN_IMAGES_PER_CLASS} images: {too_small}")
        df = df[~df["label"].isin(too_small)].reset_index(drop=True)

    return df


# --------------------------------------------------------------------------
# 2. Stratified three-way split
# --------------------------------------------------------------------------
def build_splits(
    raw_dir: Path = config.RAW_DIR,
    split_file: Path = config.SPLIT_FILE,
    force: bool = False,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """
    Create (or reuse) the stratified 70/15/15 split.

    Stratification keeps the class proportions of the full dataset inside each
    split, which matters here because PlantVillage is quite imbalanced -- some
    disease classes have a few hundred images and others several thousand.

    The split is done in two steps because scikit-learn's train_test_split
    only cuts a set in two: first train vs (val+test), then val vs test.
    """
    config.ensure_dirs()
    split_file = Path(split_file)

    if split_file.exists() and not force:
        print(f"[data] reusing existing split at {split_file} "
              "(pass --force to rebuild)")
        return load_splits(split_file)

    df = discover_images(raw_dir)

    holdout_frac = config.VAL_FRAC + config.TEST_FRAC
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_frac,
        stratify=df["label"],
        random_state=seed,
        shuffle=True,
    )

    # Of the held-out portion, split so that val and test end up the sizes
    # requested in config (0.15 / 0.15 by default -> half of the holdout each).
    test_share_of_holdout = config.TEST_FRAC / holdout_frac
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_share_of_holdout,
        stratify=holdout_df["label"],
        random_state=seed,
        shuffle=True,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    out = (
        pd.concat([train_df, val_df, test_df])
        .sort_values(["split", "label", "relpath"])
        .reset_index(drop=True)
    )
    out.to_csv(split_file, index=False)

    classes = sorted(out["label"].unique())
    config.save_classes(classes)

    print(f"[data] wrote {split_file}  ({len(out)} images, {len(classes)} classes)")
    print(out["split"].value_counts().to_string())
    return out


def load_splits(split_file: Path = config.SPLIT_FILE) -> pd.DataFrame:
    """Read the frozen split. Fails loudly rather than silently re-splitting."""
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(
            f"{split_file} not found. Run:  python src/data.py --build"
        )
    return pd.read_csv(split_file)


def get_split(split: str, split_file: Path = config.SPLIT_FILE) -> pd.DataFrame:
    """Return one of 'train' / 'val' / 'test' as a DataFrame."""
    df = load_splits(split_file)
    subset = df[df["split"] == split].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"split '{split}' is empty; valid values: "
                         f"{sorted(df['split'].unique())}")
    return subset


def subsample_per_class(
    df: pd.DataFrame,
    max_per_class: int | None,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """
    Cap the number of images per class. Used by Approach 2, where GLCM texture
    extraction over the full dataset costs hours and buys nothing.

    Passing None or 0 returns the frame untouched.
    """
    if not max_per_class:
        return df
    parts = [
        group.sample(min(len(group), max_per_class), random_state=seed)
        for _, group in df.groupby("label", sort=True)
    ]
    return (
        pd.concat(parts)
        .sort_values(["label", "relpath"])
        .reset_index(drop=True)
    )


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class image counts per split -- useful for the EDA section."""
    return (
        df.groupby(["label", "split"]).size().unstack(fill_value=0)
        .assign(total=lambda t: t.sum(axis=1))
        .sort_values("total", ascending=False)
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the shared dataset split.")
    parser.add_argument("--build", action="store_true",
                        help="create data/splits.csv and data/classes.json")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if splits.csv already exists")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--summary", action="store_true",
                        help="print the per-class distribution table")
    args = parser.parse_args()

    if args.build or args.force:
        df = build_splits(raw_dir=args.raw_dir, force=args.force)
    else:
        df = load_splits()

    if args.summary:
        table = class_distribution(df)
        print("\nPer-class distribution:")
        print(table.to_string())
        csv_path = config.METRIC_DIR / "class_distribution.csv"
        config.ensure_dirs()
        table.to_csv(csv_path)
        print(f"\n[data] saved {csv_path}")


if __name__ == "__main__":
    main()
