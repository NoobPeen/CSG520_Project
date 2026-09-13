"""
Hand-crafted feature extraction for Approach 2.

Every feature here was chosen because a plant pathologist could name what it
measures, which is the whole point of the approach: unlike a CNN filter, we
know in advance what each number is supposed to capture, so we know where the
classifier ought to be looking.

Three families, 186 numbers per image:

  colour  (114)  HSV histograms + first three moments per channel.
                 Disease shows up as chlorosis (yellowing -> hue shift),
                 necrosis (browning -> hue and saturation shift) and mottling
                 (higher variance), all of which move these statistics.

  texture  (58)  GLCM properties at two distances and four orientations,
                 plus a uniform LBP histogram. Lesions, powdery coatings and
                 speckling change local co-occurrence statistics even when
                 the average colour barely moves.

  shape    (14)  Region descriptors of the segmented leaf: circularity,
                 solidity, aspect ratio, extent, eccentricity, Hu moments.
                 Curling and marginal necrosis deform the silhouette.

All colour and texture statistics are computed over the segmented leaf only,
so the background does not leak into the features.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import skew

# skimage renamed the GLCM functions in 0.19; support both spellings.
try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except ImportError:  # pragma: no cover - older scikit-image
    from skimage.feature import (
        greycomatrix as graycomatrix,
        greycoprops as graycoprops,
        local_binary_pattern,
    )

import config

# ---- extraction settings --------------------------------------------------
HIST_BINS = 32
GLCM_DISTANCES = (1, 3)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy",
              "correlation", "ASM")
GLCM_LEVELS = 32          # grey levels are quantised: 256 levels make the
                          # co-occurrence matrix sparse and slow
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2      # uniform LBP has P+2 distinct codes
EPS = 1e-8


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def segment_leaf(bgr: np.ndarray) -> np.ndarray:
    """
    Separate leaf from background and return a uint8 mask (0 or 255).

    Primary cue is Excess Green, ExG = 2G - R - B, thresholded with Otsu.
    ExG is a standard vegetation index: it is large for green tissue and near
    zero for the grey backgrounds used in PlantVillage.

    Its weakness is exactly the case we care about -- badly necrotic leaves
    are brown, not green, so ExG can under-segment them. When the resulting
    mask covers an implausible fraction of the frame the function falls back
    to Otsu on the HSV saturation channel, where a dull grey background is
    still clearly separable from any leaf tissue, healthy or diseased.
    """
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    b, g, r = (c.astype(np.float32) for c in cv2.split(blurred))

    exg = 2.0 * g - r - b
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coverage = float(np.count_nonzero(mask)) / mask.size
    if coverage < 0.05 or coverage > 0.95:
        saturation = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)[:, :, 1]
        _, mask = cv2.threshold(saturation, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up: remove speckle, close gaps inside the blade.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = _keep_largest_component(mask)
    mask = _fill_holes(mask)

    # If segmentation collapsed entirely, fall back to the whole frame so the
    # pipeline degrades gracefully instead of dividing by zero downstream.
    if np.count_nonzero(mask) < 0.01 * mask.size:
        mask = np.full(mask.shape, 255, dtype=np.uint8)
    return mask


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """A leaf photograph has one leaf; drop every other blob."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    # Label 0 is the background component.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Necrotic patches can threshold out as holes; fill them back in."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, color=255, thickness=cv2.FILLED)
    return filled


# --------------------------------------------------------------------------
# Feature families
# --------------------------------------------------------------------------
def color_features(bgr: np.ndarray, mask: np.ndarray) -> OrderedDict:
    """HSV histograms (96) + mean/std/skew per channel in RGB and HSV (18)."""
    features: OrderedDict = OrderedDict()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # --- masked histograms, normalised so image size does not matter -------
    ranges = {"h": (0, 180), "s": (0, 256), "v": (0, 256)}
    for idx, (name, (lo, hi)) in enumerate(ranges.items()):
        hist = cv2.calcHist([hsv], [idx], mask, [HIST_BINS], [lo, hi]).flatten()
        hist = hist / (hist.sum() + EPS)
        for b in range(HIST_BINS):
            features[f"color_hist_{name}_{b:02d}"] = float(hist[b])

    # --- colour moments over leaf pixels only ------------------------------
    selection = mask > 0
    if np.count_nonzero(selection) < 50:      # degenerate mask: use everything
        selection = np.ones_like(mask, dtype=bool)

    channels = OrderedDict([
        ("B", bgr[:, :, 0]), ("G", bgr[:, :, 1]), ("R", bgr[:, :, 2]),
        ("H", hsv[:, :, 0]), ("S", hsv[:, :, 1]), ("V", hsv[:, :, 2]),
    ])
    for name, channel in channels.items():
        values = channel[selection].astype(np.float64)
        features[f"color_moment_mean_{name}"] = float(values.mean())
        features[f"color_moment_std_{name}"] = float(values.std())
        # Skewness separates "uniformly pale" from "mottled", which is often
        # the difference between a nutrient issue and a fungal infection.
        features[f"color_moment_skew_{name}"] = float(skew(values)) if values.std() > EPS else 0.0

    return features


def texture_features(bgr: np.ndarray, mask: np.ndarray) -> OrderedDict:
    """GLCM properties (48) + uniform LBP histogram (10)."""
    features: OrderedDict = OrderedDict()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Restrict the texture computation to the leaf: background pixels are set
    # to 0 so they form their own, constant grey level instead of contributing
    # spurious edges at the silhouette.
    leaf_gray = np.where(mask > 0, gray, 0)
    quantised = (leaf_gray // (256 // GLCM_LEVELS)).astype(np.uint8)

    glcm = graycomatrix(
        quantised,
        distances=list(GLCM_DISTANCES),
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)      # shape (n_distances, n_angles)
        for di, distance in enumerate(GLCM_DISTANCES):
            for ai in range(len(GLCM_ANGLES)):
                angle_deg = int(round(np.degrees(GLCM_ANGLES[ai])))
                features[f"texture_glcm_{prop}_d{distance}_a{angle_deg}"] = float(values[di, ai])

    # Local Binary Patterns: rotation-invariant micro-texture, useful for
    # powdery mildew and speckled lesions.
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    lbp_values = lbp[mask > 0] if np.count_nonzero(mask) >= 50 else lbp.flatten()
    hist, _ = np.histogram(lbp_values, bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype(np.float64) / (hist.sum() + EPS)
    for b in range(LBP_BINS):
        features[f"texture_lbp_{b:02d}"] = float(hist[b])

    return features


def shape_features(mask: np.ndarray) -> OrderedDict:
    """Silhouette descriptors of the segmented leaf (14 numbers)."""
    features: OrderedDict = OrderedDict()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        for name in ("area_ratio", "perimeter_ratio", "circularity", "aspect_ratio",
                     "extent", "solidity", "eccentricity"):
            features[f"shape_{name}"] = 0.0
        for i in range(7):
            features[f"shape_hu_{i}"] = 0.0
        return features

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    image_area = float(mask.shape[0] * mask.shape[1])
    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))

    features["shape_area_ratio"] = area / image_area
    features["shape_perimeter_ratio"] = perimeter / (2.0 * (mask.shape[0] + mask.shape[1]))
    # Circularity = 1 for a perfect disc and falls as the outline gets ragged,
    # which is what marginal necrosis and curling do to a leaf.
    features["shape_circularity"] = float(4.0 * np.pi * area / (perimeter ** 2 + EPS))
    features["shape_aspect_ratio"] = float(w) / (h + EPS)
    features["shape_extent"] = area / (w * h + EPS)
    # Solidity = area / convex-hull area: sensitive to bites, holes and lobing.
    features["shape_solidity"] = area / (hull_area + EPS)

    moments = cv2.moments(contour)
    if moments["m00"] > EPS:
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        common = np.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
        lambda1 = 0.5 * (mu20 + mu02 + common)
        lambda2 = 0.5 * (mu20 + mu02 - common)
        features["shape_eccentricity"] = float(
            np.sqrt(max(1.0 - (lambda2 / (lambda1 + EPS)), 0.0))
        )
    else:
        features["shape_eccentricity"] = 0.0

    # Hu moments are scale/rotation/translation invariant but span many orders
    # of magnitude, so the usual signed log transform is applied.
    hu = cv2.HuMoments(cv2.moments(mask)).flatten()
    for i, value in enumerate(hu):
        features[f"shape_hu_{i}"] = float(-np.sign(value) * np.log10(abs(value) + EPS))

    return features


# --------------------------------------------------------------------------
# Per-image and per-dataset extraction
# --------------------------------------------------------------------------
def extract_features(image_path: Path, img_size: int = config.A2_FEATURE_IMG_SIZE) -> OrderedDict:
    """Full 186-dimensional descriptor for one image."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image: {image_path}")

    bgr = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    mask = segment_leaf(bgr)

    features: OrderedDict = OrderedDict()
    features.update(color_features(bgr, mask))
    features.update(texture_features(bgr, mask))
    features.update(shape_features(mask))
    return features


def _safe_extract(relpath: str, root: Path, img_size: int):
    """Worker wrapper: a single unreadable file must not kill a long job."""
    try:
        return relpath, extract_features(Path(root) / relpath, img_size)
    except Exception as exc:  # noqa: BLE001 - report and skip
        print(f"[features] skipping {relpath}: {exc}")
        return relpath, None


def extract_dataset(
    df,
    root: Path = config.RAW_DIR,
    n_jobs: int = -1,
    img_size: int = config.A2_FEATURE_IMG_SIZE,
    cache_path: Path | None = None,
    force: bool = False,
):
    """
    Extract features for every row of df (columns: relpath, label).

    Results are cached as a .npz keyed on the cache_path, because extraction
    is the slow part of Approach 2 and the model search below is re-run often.

    Returns (X, y_labels, feature_names, relpaths).
    """
    from joblib import Parallel, delayed

    if cache_path is not None and Path(cache_path).exists() and not force:
        cached = np.load(cache_path, allow_pickle=True)
        print(f"[features] loaded cache {cache_path}  X={cached['X'].shape}")
        return (cached["X"], cached["y"], list(cached["names"]), list(cached["relpaths"]))

    print(f"[features] extracting {len(df)} images with n_jobs={n_jobs} ...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_safe_extract)(relpath, root, img_size) for relpath in df["relpath"]
    )

    label_by_relpath = dict(zip(df["relpath"], df["label"]))
    rows, labels, relpaths, names = [], [], [], None
    for relpath, feats in results:
        if feats is None:
            continue
        if names is None:
            names = list(feats.keys())
        rows.append([feats[k] for k in names])
        labels.append(label_by_relpath[relpath])
        relpaths.append(relpath)

    if not rows:
        raise RuntimeError("feature extraction produced no rows")

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels)
    # Guard against NaN/inf from degenerate masks before they reach sklearn.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y,
                            names=np.array(names), relpaths=np.array(relpaths))
        print(f"[features] cached -> {cache_path}")

    print(f"[features] X={X.shape}  (features per image: {len(names)})")
    return X, y, names, relpaths


def feature_group(name: str) -> str:
    """'color', 'texture' or 'shape' -- used for group-level importance."""
    return name.split("_", 1)[0]


# --------------------------------------------------------------------------
# Segmentation sanity check (worth one figure in the report)
# --------------------------------------------------------------------------
def save_segmentation_examples(df, root: Path = config.RAW_DIR,
                               n: int = 6, out_name: str = "a2_segmentation") -> Path:
    """Original / mask / masked leaf, for a random handful of images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.ensure_dirs()
    sample = df.sample(min(n, len(df)), random_state=config.SEED)

    fig, axes = plt.subplots(3, len(sample), figsize=(2.3 * len(sample), 7))
    if len(sample) == 1:
        axes = axes.reshape(3, 1)

    for col, (_, row) in enumerate(sample.iterrows()):
        bgr = cv2.imread(str(Path(root) / row["relpath"]), cv2.IMREAD_COLOR)
        bgr = cv2.resize(bgr, (config.A2_FEATURE_IMG_SIZE,) * 2, interpolation=cv2.INTER_AREA)
        mask = segment_leaf(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(str(row["label"])[:20], fontsize=7)
        axes[1, col].imshow(mask, cmap="gray")
        axes[2, col].imshow(cv2.bitwise_and(rgb, rgb, mask=mask))
        for r in range(3):
            axes[r, col].set_axis_off()

    fig.suptitle("Approach 2 segmentation: image / mask / leaf pixels used", fontsize=10)
    fig.tight_layout()
    path = config.FIG_DIR / f"{out_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[features] saved figure -> {path}")
    return path
