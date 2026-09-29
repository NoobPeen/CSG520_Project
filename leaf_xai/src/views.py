"""
Image views and perturbations shared by V1 (A2 fix), V2 and V3.

Every approach must see an image through the pipeline it was trained with:

  * A1 / A3 / A4 (deep nets)  -> "dl view": Resize(int(224*1.14)=255) + CentreCrop(224),
                                  exactly trainer.load_eval_image_uint8.
  * A2 (hand-crafted features) -> "a2 view": the WHOLE frame resized to 256x256
                                  (cv2 INTER_AREA), exactly features.extract_features.

Why the A2 view matters: A2's shape features are ratios to the frame
(shape_area_ratio = leaf area / image area, perimeter / frame perimeter). The
224 centre crop trims background, so the leaf fills ~0.59 of the crop instead
of ~0.48 of the frame and those features shift. Feeding A2 the crop dropped
its clean accuracy on the V1 sample to 0.68 (test accuracy 0.89).

V2 perturbs the ORIGINAL image and then takes each arm's own view, so a
"brightness 0.8" image is the same photograph for every arm.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

IMG_SIZE = 224
RESIZE_TO = int(IMG_SIZE * 1.14)       # 255, as in trainer.load_eval_image_uint8
A2_SIZE = 256                          # config.A2_FEATURE_IMG_SIZE


# ---------------------------------------------------------------------------
# Loading and views
# ---------------------------------------------------------------------------
def load_original(path: Path | str) -> Image.Image:
    """The photograph as stored (PlantVillage: 256x256 RGB)."""
    return Image.open(path).convert("RGB")


def _resize_short_side(img: Image.Image, short: int) -> Image.Image:
    """torchvision.transforms.Resize(short) on a PIL image (bilinear, antialiased)."""
    try:
        from torchvision import transforms
        return transforms.Resize(short)(img)
    except ImportError:                         # CPU-only box without torchvision
        w, h = img.size
        if w <= h:
            size = (short, int(short * h / w))
        else:
            size = (int(short * w / h), short)
        return img.resize(size, Image.BILINEAR)


def centre_offsets(h: int, w: int, crop: int = IMG_SIZE) -> tuple[int, int]:
    """(top, left) used by torchvision CenterCrop."""
    return int(round((h - crop) / 2.0)), int(round((w - crop) / 2.0))


def eval_canvas(img: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """Resize(1.14 x size) without the crop: the frame the 224 crop lives in."""
    return np.array(_resize_short_side(img, int(img_size * 1.14)), dtype=np.uint8)


def dl_view(img: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """uint8 HxWx3 exactly as the deep nets are evaluated."""
    canvas = eval_canvas(img, img_size)
    top, left = centre_offsets(*canvas.shape[:2], img_size)
    return canvas[top:top + img_size, left:left + img_size].copy()


def a2_view(img: Image.Image | np.ndarray, size: int = A2_SIZE) -> np.ndarray:
    """uint8 RGB size x size, whole frame, as A2 was trained (cv2 INTER_AREA)."""
    arr = np.asarray(img, dtype=np.uint8)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def paste_crop(canvas: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """Put a (possibly perturbed) centre crop back into its frame."""
    out = canvas.copy()
    top, left = centre_offsets(*canvas.shape[:2], crop.shape[0])
    out[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return out


# ---------------------------------------------------------------------------
# V2 perturbations (applied to the original photograph)
# ---------------------------------------------------------------------------
# Severities deliberately bracket the training augmentation: the mildest level
# sits inside a typical ColorJitter(0.2)/RandomRotation(15) range, the harsher
# levels sit outside it. Check torch_data.build_transforms and state the actual
# training ranges next to the V2 table.
PERTURBATIONS: dict[str, tuple[float, ...]] = {
    "brightness": (0.6, 0.8, 1.2, 1.4),     # PIL factor, 1 = unchanged
    "contrast":   (0.6, 0.8, 1.2, 1.4),     # PIL factor, 1 = unchanged
    "rotation":   (-30.0, -15.0, 15.0, 30.0),  # degrees, reflect-padded
    "blur":       (1.0, 2.0, 3.0),          # Gaussian sigma in pixels (256-px frame)
}


def conditions() -> list[tuple[str, float]]:
    """[('clean', 0), ('brightness', 0.6), ...] in a fixed order."""
    out = [("clean", 0.0)]
    for kind, levels in PERTURBATIONS.items():
        out += [(kind, float(v)) for v in levels]
    return out


def condition_name(kind: str, level: float) -> str:
    return "clean" if kind == "clean" else f"{kind}_{level:g}"


def perturb(img: Image.Image, kind: str, level: float) -> Image.Image:
    if kind == "clean":
        return img
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(level)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(level)
    arr = np.asarray(img, dtype=np.uint8)
    if kind == "rotation":
        h, w = arr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), level, 1.0)
        # Reflect padding: black corners would be a new, out-of-distribution cue.
        out = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(out)
    if kind == "blur":
        return Image.fromarray(cv2.GaussianBlur(arr, (0, 0), sigmaX=level, sigmaY=level))
    raise ValueError(f"unknown perturbation {kind!r}")
