"""
V1 -- "Where does the model look?"  Occlusion + LIME + leaf containment,
computed the same way for every approach.

One interface for every arm: predict_proba(uint8 N x 224 x 224 x 3) -> (N, C).
A1 (ResNet50 / EfficientNet-B0, incl. the scratch ablation), A2 (hand-crafted
features + RF/SVM), A3 (custom CNN) and A4 (ViT) are all wrapped to it, so
the perturbations, the images, the masks and the scoring are identical and
only the model changes.

------------------------------------------------------------------------------
Protocol (agreed in method-and-metrics; do not change without re-running all arms)
------------------------------------------------------------------------------
Images   Stratified sample of the frozen TEST split (fixed seed): k_occ images
         per class for occlusion, and the first k_lime of those for LIME.
         The same images are used for every arm. Each image gets the evaluation
         geometry (resize to 255, centre-crop 224), trainer.load_eval_image_uint8.

Masks    Ground-truth leaf masks from PlantVillage's segmented variant
         (<name>_final_masked.jpg), with the same resize + crop. Models are
         never trained on the segmented images; they are only used for masks.
         If a mask is missing the ExG segmentation from features.py is used
         and the row is flagged (mask_source = "exg").

Target   The class the model predicts on the clean image: we explain the
         decision it actually made.

Occlusion  Patch 32, stride 32 -> a 7 x 7 grid (49 positions). Each patch is
         filled with the dataset mean colour (or a blur, --fill blur), never
         black or grey. Importance of a patch = p_target(clean) -
         p_target(occluded).

LIME     lime_image with num_samples=200, hide_color = dataset mean RGB, and
         quickshift superpixels (kernel 4, max_dist 200, ratio 0.2). The
         superpixels are computed once per image and reused for every arm.
         The superpixel weights are painted back onto the pixels.

Containment  For a map A (pixels) and binary leaf mask M:
             score = sum(max(A,0) * M) / sum(max(A,0))
         i.e. the share of positive evidence that falls on the leaf. Occlusion
         patches are painted at full resolution first, which is the same as
         weighting each patch by its fractional overlap with the leaf (soft
         assignment). A model with no leaf preference (uniform evidence)
         scores exactly the image's own leaf fraction f, which is the per-image
         null. Reported per image:
             lift_ratio = score / f          (1 = chance, >1 prefers the leaf)
             lift_kappa = (score - f)/(1 - f) (0 = chance, 1 = all evidence on the leaf,
                                               <0 = prefers background)
         The corpus-level null (mean f; ~0.477 for PlantVillage) is printed as
         the headline reference.

Agreement  Occlusion map upsampled to 224 (nearest -- the patch values are
         piecewise constant by construction), then both maps averaged inside
         each LIME superpixel, Spearman rank correlation over the superpixels.
         Signed values are used here (not clipped), so it compares the full
         ranking of evidence.

Outputs  <out>/explain/<arm>/per_image.csv, summary.json, occlusion.npz, lime.npz,
         examples.png, and <out>/explain/v1_table.csv after --summarize.
         Progress is checkpointed after every image, so a killed run resumes.

    python src/explain.py --arms a1_resnet50 --mask-root <.../segmented>
    python src/explain.py --arms a2_rf --n-jobs -1 --lime-per-class 2 --mask-root ...
    python src/explain.py --summarize
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

import config
import data as data_module
import trainer

ARMS = ("a1_resnet50", "a1_resnet50_scratch", "a1_efficientnet_b0",
        "a2_rf", "a2_svm", "a3_cnn", "a4_vit")


# ==========================================================================
# 1. Model wrappers -- every arm becomes predict_proba(uint8 NxHxWx3) -> NxC
# ==========================================================================
class A2Wrapper:
    """
    Hand-crafted features + sklearn pipeline.

    The features are recomputed on every perturbed image, INCLUDING the leaf
    segmentation, so a patch that changes what ExG calls "leaf" also changes
    the shape features. That is the honest causal behaviour of A2.

    Full-frame view (fix of 21 Sep, revised). A2 was trained on the WHOLE
    frame read with cv2 and resized to 256 x 256 (features.extract_features).
    Two things break it: the 224 centre crop changes the frame-relative shape
    features (clean sample accuracy 0.68), and resampling the photo (256 -> 255
    -> 256) blurs the fine texture that GLCM/LBP read (0.52 when the crop was
    pasted back into a resampled frame). So for A2 nothing is resampled:
    set_context(path) keeps the training-exact 256 frame and the clean 224
    crop. For each crop explain.py asks about, only the pixels that differ from
    the clean crop (occlusion patch, hidden LIME superpixels) are mapped into
    the 256 frame; every other pixel is the original. A clean crop therefore
    gives exactly the training features. Masks and scoring stay on the crop.
    With no context set (V2 passes whole frames) images are used as given.
    """

    def __init__(self, model, classes: list[str], n_jobs: int = -1) -> None:
        self.model = model
        self.classes = list(classes)
        self.n_jobs = n_jobs
        est = getattr(model, "steps", [[None, model]])[-1][1]
        if hasattr(est, "n_jobs"):
            est.n_jobs = n_jobs              # parallel tree prediction
        self.model_classes = np.asarray(getattr(model, "classes_", np.arange(len(classes))))
        self.frame = None                    # training-exact 256 frame of the image being explained

    def set_context(self, path, img_size: int = config.IMG_SIZE) -> None:
        """Remember the image the next 224 crops come from (see class docstring)."""
        from PIL import Image
        self.frame = a2_training_frame(path)
        self.clean_crop = trainer.load_eval_image_uint8(path, img_size)
        w, h = Image.open(path).size                  # torchvision Resize(short side) geometry
        short = int(img_size * 1.14)
        self.eval_hw = (int(short * h / w), short) if w <= h else (short, int(short * w / h))
        self.crop_tl = (int(round((self.eval_hw[0] - img_size) / 2.0)),
                        int(round((self.eval_hw[1] - img_size) / 2.0)))

    def clear_context(self) -> None:
        self.frame = None

    def _to_frame(self, crop: np.ndarray) -> np.ndarray:
        """Perturbed 224 crop -> 256 training frame, copying only the changed pixels."""
        import cv2
        changed = np.any(crop != self.clean_crop, axis=-1)
        if not changed.any():
            return self.frame
        (hc, wc), (top, left), n = self.eval_hw, self.crop_tl, crop.shape[0]
        mask = np.zeros((hc, wc), np.uint8)
        vals = np.zeros((hc, wc, 3), np.uint8)
        mask[top:top + n, left:left + n] = changed
        vals[top:top + n, left:left + n] = crop
        H, W = self.frame.shape[:2]
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        vals = cv2.resize(vals, (W, H), interpolation=cv2.INTER_NEAREST)
        out = self.frame.copy()
        out[mask] = vals[mask]
        return out

    def proba_from_features(self, rows) -> np.ndarray:
        X = np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        raw = self.model.predict_proba(X)
        out = np.zeros((len(X), len(self.classes)), dtype=np.float32)
        out[:, self.model_classes.astype(int)] = raw   # sklearn orders columns by classes_
        return out

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        from joblib import Parallel, delayed
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[None]
        images = np.clip(images, 0, 255).astype(np.uint8)
        if self.frame is not None and images.shape[1:3] == self.clean_crop.shape[:2]:
            images = np.stack([self._to_frame(im) for im in images])
        if len(images) > 1 and self.n_jobs != 1:
            rows = Parallel(n_jobs=self.n_jobs)(delayed(_a2_features)(im) for im in images)
        else:
            rows = [_a2_features(im) for im in images]
        return self.proba_from_features(rows)

    __call__ = predict_proba


def a2_training_frame(path) -> np.ndarray:
    """RGB uint8 frame exactly as features.extract_features builds it (cv2 read + INTER_AREA)."""
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image: {path}")
    size = getattr(config, "A2_FEATURE_IMG_SIZE", 256)
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _a2_features(rgb: np.ndarray) -> list[float]:
    import cv2
    import features as F
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = F.segment_leaf(bgr)
    feats = {}
    feats.update(F.color_features(bgr, mask))
    feats.update(F.texture_features(bgr, mask))
    feats.update(F.shape_features(mask))
    return list(feats.values())


class _UInt8Guard:
    """LIME hands over float copies; cast back so every arm sees uint8."""

    def __init__(self, predictor) -> None:
        self.predictor = predictor

    def __call__(self, images):
        return self.predictor.predict_proba(np.clip(np.asarray(images), 0, 255).astype(np.uint8))


def _fp32(predictor):
    """
    Explanations run in full fp32. Under fp16 autocast the logits carry
    rounding noise of ~1e-3 in probability, which is the same size as most
    single-patch occlusion drops on these near-saturated models.
    """
    predictor.use_amp = False
    return predictor


def load_arm(arm: str, ckpt_dir: Path, classes: list[str], device=None, n_jobs: int = -1):
    """Build the predict_proba wrapper for one arm from its saved checkpoint."""
    import torch
    ckpt_dir = Path(ckpt_dir)
    if arm.startswith("a1_"):
        import approach1_transfer as a1
        arch = arm[3:].replace("_scratch", "")
        path = ckpt_dir / f"{arm}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model = a1.build_model(arch, num_classes=len(classes), pretrained=False)
        model.load_state_dict(state)
        return trainer.ProbaWrapper(model, config.IMAGENET_MEAN, config.IMAGENET_STD,
                                    classes, device=device, batch_size=64, use_amp=False)
    if arm.startswith("a2_"):
        from joblib import load
        path = ckpt_dir / f"{arm}.joblib"
        print(f"[explain] loading {path.name} (the RF is large; this can take a minute)")
        return A2Wrapper(load(path), classes, n_jobs=n_jobs)
    if arm == "a3_cnn":
        import approach3_customcnn as a3
        return _fp32(a3.load_predictor(ckpt_dir / "a3_cnn.pt", device=device))
    if arm == "a4_vit":
        import approach4_vit as a4
        return _fp32(a4.load_predictor(ckpt_dir / "a4_vit.pt", device=device))
    raise ValueError(f"unknown arm '{arm}'; choose from {ARMS}")


# ==========================================================================
# 2. Images, sample, masks, fill colour
# ==========================================================================
def select_sample(split_file: Path, k_occ: int, k_lime: int, seed: int) -> pd.DataFrame:
    """
    k_occ test images per class (fixed seed); the first k_lime of each class
    are flagged for LIME. Nested, so LIME images are a subset of the occlusion
    images and every arm sees exactly the same rows.
    """
    test = data_module.get_split("test", split_file)
    rng = np.random.default_rng(seed)
    parts = []
    for label, group in test.groupby("label", sort=True):
        idx = rng.permutation(len(group))[:k_occ]
        g = group.iloc[idx].copy()
        g["lime"] = np.arange(len(g)) < k_lime
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)


def dataset_mean_rgb(split_file: Path, data_root: Path, cache: Path,
                     n: int = 600, seed: int = config.SEED) -> tuple[float, float, float]:
    """Mean RGB over n training images at evaluation geometry (cached)."""
    if cache.exists():
        return tuple(json.loads(cache.read_text())["mean_rgb"])
    train = data_module.get_split("train", split_file)
    rows = train.sample(min(n, len(train)), random_state=seed)
    acc = np.zeros(3)
    for rel in rows["relpath"]:
        acc += trainer.load_eval_image_uint8(Path(data_root) / rel).reshape(-1, 3).mean(0)
    mean = tuple(float(v) for v in acc / len(rows))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mean_rgb": mean, "n_images": len(rows)}, indent=2))
    return mean


class MaskIndex:
    """
    Leaf masks from PlantVillage's segmented variant.

    color/<class>/<name>.JPG  <->  segmented/<class>/<name>_final_masked.jpg
    The segmented image has a black background: mask = pixels that are not
    near-black, cleaned (open, largest component, fill holes), then given the
    same resize + centre-crop as the model input.
    """

    def __init__(self, mask_root: Path | None, img_size: int = config.IMG_SIZE,
                 thresh: int = 20) -> None:
        self.img_size, self.thresh = img_size, thresh
        self.index: dict[tuple[str, str], Path] = {}
        if mask_root is not None and Path(mask_root).exists():
            for p in Path(mask_root).rglob("*"):
                if p.is_file() and "_final_masked" in p.name:
                    stem = p.stem.replace("_final_masked", "").lower()
                    self.index[(p.parent.name, stem)] = p
        print(f"[explain] mask index: {len(self.index)} segmented images"
              + ("" if self.index else "  (none -- falling back to ExG masks)"))

    def get(self, relpath: str, rgb224: np.ndarray) -> tuple[np.ndarray, str]:
        cls, fname = relpath.split("/")[-2:]
        path = self.index.get((cls, Path(fname).stem.lower()))
        if path is not None:
            return self._from_segmented(path), "gt"
        return self._exg(rgb224), "exg"

    def _from_segmented(self, path: Path) -> np.ndarray:
        import cv2
        import features as F
        seg = np.asarray(Image.open(path).convert("RGB"))
        m = (seg.max(axis=2) > self.thresh).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = F._fill_holes(F._keep_largest_component(m))
        s = self.img_size
        pil = Image.fromarray(m).resize(self._resized_size(m.shape, int(s * 1.14)),
                                        Image.NEAREST)
        w, h = pil.size
        left, top = int(round((w - s) / 2.0)), int(round((h - s) / 2.0))  # = CenterCrop
        crop = np.asarray(pil.crop((left, top, left + s, top + s)))
        return (crop > 127).astype(np.float32)

    @staticmethod
    def _resized_size(shape, short: int) -> tuple[int, int]:
        """(w, h) that torchvision.Resize(short) produces (shorter side -> short)."""
        h, w = shape[:2]
        if w <= h:
            return short, int(short * h / w)
        return int(short * w / h), short

    @staticmethod
    def _exg(rgb224: np.ndarray) -> np.ndarray:
        import cv2
        import features as F
        return (F.segment_leaf(cv2.cvtColor(rgb224, cv2.COLOR_RGB2BGR)) > 0).astype(np.float32)


# ==========================================================================
# 3. Attribution methods
# ==========================================================================
def occlusion_map(predict, img: np.ndarray, target: int, p_clean: float,
                  patch: int, fill: str, mean_rgb) -> np.ndarray:
    """(G x G) drop in p_target when each patch is replaced; G = 224 / patch."""
    import cv2
    H, W = img.shape[:2]
    gh, gw = H // patch, W // patch
    filler = (cv2.GaussianBlur(img, (0, 0), sigmaX=10) if fill == "blur"
              else np.broadcast_to(np.array(mean_rgb, dtype=np.float32).round().astype(np.uint8),
                                   img.shape))
    batch = np.repeat(img[None], gh * gw, axis=0)
    for i in range(gh):
        for j in range(gw):
            ys, xs = slice(i * patch, (i + 1) * patch), slice(j * patch, (j + 1) * patch)
            batch[i * gw + j, ys, xs] = filler[ys, xs]
    probs = predict(batch)[:, target]
    return (p_clean - probs).reshape(gh, gw).astype(np.float32)


def quickshift_segments(img: np.ndarray, seed: int) -> np.ndarray:
    """LIME's default superpixels, computed once per image and shared by all arms."""
    from skimage.segmentation import quickshift
    try:                                   # scikit-image >= 0.22
        return quickshift(img, kernel_size=4, max_dist=200, ratio=0.2, rng=seed)
    except TypeError:                      # older releases
        return quickshift(img, kernel_size=4, max_dist=200, ratio=0.2, random_seed=seed)


def lime_weights(predict, img: np.ndarray, target: int, segments: np.ndarray,
                 mean_rgb, num_samples: int, seed: int) -> tuple[np.ndarray, float]:
    """Per-superpixel LIME weights for `target`, plus the surrogate's R^2."""
    from lime import lime_image
    import inspect
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    kwargs = dict(labels=(target,), top_labels=None, hide_color=tuple(mean_rgb),
                  num_features=100000, num_samples=num_samples, batch_size=50,
                  segmentation_fn=lambda _: segments, random_seed=seed)
    if "progress_bar" in inspect.signature(explainer.explain_instance).parameters:
        kwargs["progress_bar"] = False          # newer lime releases only
    elif hasattr(lime_image, "tqdm"):
        lime_image.tqdm = lambda it, *a, **k: it  # older releases: silence the per-image bar
    exp = explainer.explain_instance(img, _UInt8Guard(predict), **kwargs)
    w = np.zeros(int(segments.max()) + 1, dtype=np.float32)
    for seg_id, weight in exp.local_exp[target]:
        w[seg_id] = weight
    return w, float(getattr(exp, "score", np.nan))


# ==========================================================================
# 4. Scoring
# ==========================================================================
def upsample_nearest(grid: np.ndarray, size: int) -> np.ndarray:
    k = size // grid.shape[0]
    return np.kron(grid, np.ones((k, k), dtype=grid.dtype))


def containment(attr: np.ndarray, mask: np.ndarray) -> float:
    """Share of positive evidence on the leaf; NaN if there is no positive evidence."""
    pos = np.clip(attr, 0, None)
    total = pos.sum()
    if total <= 1e-12:
        return float("nan")
    return float((pos * mask).sum() / total)


MAX_LEAF_FRACTION = 0.9   # overwritten from --max-leaf-fraction


def lifts(score: float, f: float) -> tuple[float, float]:
    """
    Chance-corrected containment. Undefined (NaN) when the leaf fills more
    than MAX_LEAF_FRACTION of the crop: with almost no background left,
    (1 - f) is tiny and one noisy background patch sends the kappa form to
    -4 or below (seen on Corn___healthy), which would dominate any mean.
    """
    if np.isnan(score) or f <= 0 or f > MAX_LEAF_FRACTION:
        return float("nan"), float("nan")
    return score / f, (score - f) / (1 - f)


def map_agreement(occ_grid: np.ndarray, lime_w: np.ndarray, segments: np.ndarray) -> float:
    occ_pix = upsample_nearest(occ_grid, segments.shape[0])
    ids = np.unique(segments)
    occ_avg = np.array([occ_pix[segments == s].mean() for s in ids])
    lime_avg = lime_w[ids]
    if np.ptp(occ_avg) == 0 or np.ptp(lime_avg) == 0:
        return float("nan")
    return float(spearmanr(occ_avg, lime_avg).correlation)


# ==========================================================================
# 5. Run one arm (resumable)
# ==========================================================================
def _save_state(state: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh)
    os.replace(tmp, path)


def run_arm(arm: str, args, sample: pd.DataFrame, classes, mean_rgb, masks: MaskIndex,
            out_root: Path) -> dict:
    arm_dir = out_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    state_path = arm_dir / "state.pkl"
    state = {"occ": {}, "lime": {}, "meta": {}}
    if state_path.exists() and not args.fresh:
        state = pickle.loads(state_path.read_bytes())
        print(f"[{arm}] resuming: {len(state['occ'])} occlusion / {len(state['lime'])} LIME done")

    # A2 is CPU-bound: fewer LIME images, as agreed.
    k_lime = args.lime_per_class_a2 if arm.startswith("a2_") else args.lime_per_class
    rows = sample.copy()
    rows["lime"] = rows.groupby("label").cumcount() < k_lime
    if args.limit:
        rows = rows.head(args.limit)

    if args.rescore:
        print(f"[{arm}] --rescore: recomputing scores from saved maps (no model run)")
        return score_arm(arm, rows, state, classes, masks, arm_dir, args)

    predict = load_arm(arm, args.ckpt_dir, classes, device=args.device, n_jobs=args.n_jobs)
    seg_cache = out_root / "_segments"
    seg_cache.mkdir(parents=True, exist_ok=True)

    started, n_done_before = time.perf_counter(), len(state["occ"])
    for n, row in enumerate(rows.itertuples(index=False), start=1):
        rel = row.relpath
        need_occ = "occlusion" in args.methods and rel not in state["occ"]
        need_lime = "lime" in args.methods and row.lime and rel not in state["lime"]
        if not (need_occ or need_lime):
            continue
        img = trainer.load_eval_image_uint8(Path(args.data_root) / rel, args.img_size)
        if hasattr(predict, "set_context"):          # A2: features on the full frame
            predict.set_context(Path(args.data_root) / rel, args.img_size)
        probs = predict.predict_proba(img[None])[0]
        target = int(probs.argmax())
        base = {"p_target": float(probs[target]), "target": target,
                "true": classes.index(row.label)}

        if need_occ:
            grid = occlusion_map(predict.predict_proba, img, target, base["p_target"],
                                 args.patch, args.fill, mean_rgb)
            state["occ"][rel] = {**base, "grid": grid}
        if need_lime:
            seg_file = seg_cache / (rel.replace("/", "__") + ".npy")
            if seg_file.exists():
                segments = np.load(seg_file)
            else:
                segments = quickshift_segments(img, args.seed).astype(np.int32)
                tmp = seg_file.with_name(seg_file.name + f".{os.getpid()}.tmp")
                with open(tmp, "wb") as fh:          # atomic: two runtimes may share the cache
                    np.save(fh, segments)
                os.replace(tmp, seg_file)
            w, r2 = lime_weights(predict, img, target, segments, mean_rgb,
                                 args.lime_samples, args.seed)
            state["lime"][rel] = {**base, "weights": w, "r2": r2}

        _save_state(state, state_path)
        done = len(state["occ"]) - n_done_before
        if n % 10 == 0 or n == len(rows):
            rate = (time.perf_counter() - started) / max(done, 1)
            left = sum(1 for r in rows.itertuples() if r.relpath not in state["occ"])
            print(f"[{arm}] {n}/{len(rows)} images  "
                  f"({rate:.1f}s/img, ~{left * rate / 60:.0f} min left)", flush=True)

    return score_arm(arm, rows, state, classes, masks, arm_dir, args)


def check_accuracy(arm: str, args, sample: pd.DataFrame, classes) -> float:
    """
    --check-accuracy: clean accuracy on the sample only (no maps). For A2 it also
    prints diagnostics: accuracy on the bare 224 crop (old behaviour), whether
    re-extracted features match the cached training-time features, the RF's
    accuracy on those cached rows, and whether PIL and cv2 decode identically.
    """
    rows = sample.head(args.limit) if args.limit else sample
    predict = load_arm(arm, args.ckpt_dir, classes, device=args.device, n_jobs=args.n_jobs)
    is_a2 = hasattr(predict, "set_context")
    correct, correct_crop, feats, decode_diff = [], [], {}, []
    for row in rows.itertuples(index=False):
        path = Path(args.data_root) / row.relpath
        y = classes.index(row.label)
        img = trainer.load_eval_image_uint8(path, args.img_size)
        if is_a2:
            predict.set_context(path, args.img_size)
        correct.append(int(predict.predict_proba(img[None])[0].argmax()) == y)
        if is_a2:
            predict.clear_context()
            correct_crop.append(int(predict.predict_proba(img[None])[0].argmax()) == y)
            frame = a2_training_frame(path)
            feats[row.relpath] = _a2_features(frame)
            from PIL import Image
            pil = np.asarray(Image.open(path).convert("RGB"))
            if pil.shape == frame.shape:
                decode_diff.append(int(np.abs(pil.astype(int) - frame.astype(int)).max()))
    acc = float(np.mean(correct))
    print(f"[{arm}] clean accuracy on {len(correct)} sample images: {acc:.3f}"
          + ("  (A2 test accuracy is 0.89)" if is_a2 else ""))
    if is_a2:
        print(f"[{arm}] diagnostics:")
        print(f"  bare 224 crop (old behaviour)       : {np.mean(correct_crop):.3f}")
        if decode_diff:
            print(f"  PIL vs cv2 decode, max pixel diff   : median {np.median(decode_diff):.0f}, "
                  f"max {np.max(decode_diff)}  (0 = identical)")
        cache = (Path(config.OUTPUT_DIR) / "features" /
                 f"a2_test_n400_s{getattr(config, 'A2_FEATURE_IMG_SIZE', 256)}.npz")
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            index = {str(r): i for i, r in enumerate(z["relpaths"])}
            common = [r for r in feats if r in index]
            if common:
                Xc = np.nan_to_num(z["X"][[index[r] for r in common]].astype(np.float32))
                Xn = np.nan_to_num(np.asarray([feats[r] for r in common], dtype=np.float32))
                diff = np.abs(Xc - Xn).max(1)
                ys = np.array([classes.index(str(z["y"][index[r]])) for r in common])
                acc_cache = float((predict.proba_from_features(Xc).argmax(1) == ys).mean())
                acc_new = float((predict.proba_from_features(Xn).argmax(1) == ys).mean())
                worst = np.abs(Xc - Xn).max(0)
                print(f"  sample images also in the A2 cache  : {len(common)}")
                print(f"  RF accuracy on CACHED features      : {acc_cache:.3f}")
                print(f"  RF accuracy on RE-EXTRACTED features: {acc_new:.3f}")
                print(f"  feature max |diff| per image        : median {np.median(diff):.2e}, "
                      f"max {diff.max():.2e}  (~0 = features reproduce)")
                names = [str(n) for n in z["names"]] if "names" in z else [str(i) for i in range(len(worst))]
                top = np.argsort(-worst)[:5]
                print("  largest differing features          : "
                      + ", ".join(f"{names[i]}={worst[i]:.3g}" for i in top))
        else:
            print(f"  (no feature cache at {cache}; skipped the cache comparison)")
    return acc


def score_arm(arm, rows, state, classes, masks: MaskIndex, arm_dir: Path, args) -> dict:
    """Containment, lifts and agreement per image -> CSV, summary, figure, npz."""
    records, mask_cache = [], {}
    seg_cache = arm_dir.parent / "_segments"
    for row in rows.itertuples(index=False):
        rel = row.relpath
        occ, lim = state["occ"].get(rel), state["lime"].get(rel)
        if occ is None and lim is None:
            continue
        img = trainer.load_eval_image_uint8(Path(args.data_root) / rel, args.img_size)
        mask, src = masks.get(rel, img)
        mask_cache[rel] = mask
        f = float(mask.mean())
        info = occ or lim
        rec = {"relpath": rel, "label": row.label, "true": info["true"],
               "pred": info["target"], "correct": info["true"] == info["target"],
               "p_target": info["p_target"], "leaf_fraction": f, "mask_source": src}
        if occ is not None:
            s = containment(upsample_nearest(occ["grid"], mask.shape[0]), mask)
            rec["occ_score"] = s
            rec["occ_lift_ratio"], rec["occ_lift_kappa"] = lifts(s, f)
            rec["occ_max_drop"] = float(occ["grid"].max())
            # Same occlusion run on the log-probability scale: with label
            # smoothing and softmax saturation, probability drops are
            # compressed near p ~ 0.95; log p_target spreads them out.
            p0 = max(occ["p_target"], 1e-6)
            dlog = np.log(p0) - np.log(np.clip(p0 - occ["grid"], 1e-6, None))
            s = containment(upsample_nearest(dlog, mask.shape[0]), mask)
            rec["occ_logp_score"] = s
            rec["occ_logp_lift_ratio"], rec["occ_logp_lift_kappa"] = lifts(s, f)
        if lim is not None:
            segments = np.load(seg_cache / (rel.replace("/", "__") + ".npy"))
            s = containment(lim["weights"][segments], mask)
            rec["lime_score"] = s
            rec["lime_lift_ratio"], rec["lime_lift_kappa"] = lifts(s, f)
            rec["lime_r2"] = lim["r2"]
            if occ is not None:
                rec["spearman_occ_lime"] = map_agreement(occ["grid"], lim["weights"], segments)
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    df.to_csv(arm_dir / "per_image.csv", index=False)

    # Raw maps, so figures/stats can be redone without re-running the model.
    occ_keys = [r for r in df["relpath"] if r in state["occ"]]
    if occ_keys:
        np.savez_compressed(arm_dir / "occlusion.npz", relpaths=np.array(occ_keys),
                            grids=np.stack([state["occ"][r]["grid"] for r in occ_keys]),
                            targets=np.array([state["occ"][r]["target"] for r in occ_keys]))
    lime_keys = [r for r in df["relpath"] if r in state["lime"]]
    if lime_keys:
        np.savez_compressed(arm_dir / "lime.npz", relpaths=np.array(lime_keys),
                            weights=np.array([state["lime"][r]["weights"] for r in lime_keys],
                                             dtype=object),
                            targets=np.array([state["lime"][r]["target"] for r in lime_keys]))

    summary = summarize_df(arm, df, args)
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    try:
        save_examples(arm, df, state, mask_cache, seg_cache, arm_dir, args)
    except Exception as exc:  # a figure must never lose a night of compute
        print(f"[{arm}] figure skipped: {exc}")
    return summary


# ==========================================================================
# 6. Summaries
# ==========================================================================
SIGNAL_DROP = 0.05   # an occlusion map "has signal" if some patch moves p_target by >= 5 points

METRICS = ("occ_score", "occ_lift_ratio", "occ_lift_kappa",
           "occ_logp_score", "occ_logp_lift_kappa",
           "lime_score", "lime_lift_ratio", "lime_lift_kappa",
           "spearman_occ_lime", "lime_r2")


def _ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) < 2:
        return float("nan"), float("nan")
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize_df(arm: str, df: pd.DataFrame, args) -> dict:
    out = {"arm": arm, "n_images": int(len(df)),
           "sample_accuracy": float(df["correct"].mean()),
           "mean_leaf_fraction": float(df["leaf_fraction"].mean()),
           "mask_source_counts": df["mask_source"].value_counts().to_dict(),
           "fill": args.fill, "patch": args.patch, "lime_samples": args.lime_samples,
           "metrics": {}}
    for m in METRICS:
        if m not in df:
            continue
        for subset, d in (("all", df), ("correct", df[df["correct"]])):
            v = d[m].dropna().to_numpy(dtype=float)
            if len(v) == 0:
                continue
            lo, hi = _ci(v, args.seed)
            out["metrics"].setdefault(m, {})[subset] = {
                "n": int(len(v)), "n_nan": int(d[m].isna().sum()),
                "mean": float(v.mean()), "median": float(np.median(v)),
                "ci95": [lo, hi]}
    if "occ_max_drop" in df:
        strong = df[df["occ_max_drop"] >= SIGNAL_DROP]
        out["occ_signal_share"] = float(len(strong) / max(len(df), 1))
        out["occ_max_drop_median"] = float(df["occ_max_drop"].median())
        v = strong["occ_lift_kappa"].dropna().to_numpy(dtype=float)
        if len(v):
            lo, hi = _ci(v, args.seed)
            out["metrics"]["occ_lift_kappa_signal_only"] = {"all": {
                "n": int(len(v)), "n_nan": 0, "mean": float(v.mean()),
                "median": float(np.median(v)), "ci95": [lo, hi]}}
    for method in ("occ", "lime"):
        col = f"{method}_score"
        if col in df:
            d = df.dropna(subset=[col])
            out[f"{method}_share_above_chance"] = float((d[col] > d["leaf_fraction"]).mean())
    return out


def print_summary(s: dict) -> None:
    print(f"\n=== V1 / {s['arm']} ===  images={s['n_images']}  "
          f"sample acc={s['sample_accuracy']:.3f}  mean leaf fraction (null)="
          f"{s['mean_leaf_fraction']:.3f}  masks={s['mask_source_counts']}")
    for m, d in s["metrics"].items():
        a = d.get("all")
        if a:
            print(f"  {m:<20} mean {a['mean']:.3f}  median {a['median']:.3f}  "
                  f"95% CI [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]  n={a['n']}")
    for k in ("occ_signal_share", "occ_max_drop_median",
              "occ_share_above_chance", "lime_share_above_chance"):
        if k in s:
            print(f"  {k:<24} {s[k]:.3f}")


def summarize_all(out_root: Path) -> pd.DataFrame:
    """One row per arm: the V1 table for the report."""
    rows = []
    for summ in sorted(out_root.glob("*/summary.json")):
        s = json.loads(summ.read_text())
        row = {"arm": s["arm"], "n_images": s["n_images"],
               "sample_acc": s["sample_accuracy"], "null_leaf_fraction": s["mean_leaf_fraction"]}
        for m in (*METRICS, "occ_lift_kappa_signal_only"):
            a = s["metrics"].get(m, {}).get("all")
            if a:
                row[m] = a["mean"]
                row[f"{m}_ci"] = f"[{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]"
        row["occ_signal_share"] = s.get("occ_signal_share")
        row["occ_share_above_chance"] = s.get("occ_share_above_chance")
        row["lime_share_above_chance"] = s.get("lime_share_above_chance")
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(out_root / "v1_table.csv", index=False)
        cols = [c for c in ("arm", "sample_acc", "null_leaf_fraction", "occ_signal_share",
                            "occ_lift_kappa", "occ_lift_kappa_signal_only",
                            "occ_logp_lift_kappa", "lime_lift_kappa",
                            "spearman_occ_lime") if c in table]
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n[explain] wrote {out_root / 'v1_table.csv'}")
    return table


# ==========================================================================
# 7. Figure: image + mask outline | occlusion | LIME, for 6 classes
# ==========================================================================
def save_examples(arm, df, state, mask_cache, seg_cache, arm_dir, args, n: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    have = df[df["relpath"].isin(state["lime"].keys()) & df["relpath"].isin(state["occ"].keys())]
    if have.empty:
        have = df
    picks = have.groupby("label").head(1).head(n)
    fig, axes = plt.subplots(3, len(picks), figsize=(2.5 * len(picks), 7.8), squeeze=False)
    for j, r in enumerate(picks.itertuples(index=False)):
        img = trainer.load_eval_image_uint8(Path(args.data_root) / r.relpath, args.img_size)
        mask = mask_cache[r.relpath]
        axes[0, j].imshow(img)
        axes[0, j].contour(mask, levels=[0.5], colors="yellow", linewidths=1)
        axes[0, j].set_title(f"{r.label[:22]}\nleaf f={r.leaf_fraction:.2f}", fontsize=7,
                             color="green" if r.correct else "red")
        if r.relpath in state["occ"]:
            grid = state["occ"][r.relpath]["grid"]
            axes[1, j].imshow(img)
            axes[1, j].imshow(zoom(np.clip(grid, 0, None), img.shape[0] / grid.shape[0], order=1),
                              cmap="jet", alpha=0.45)
            axes[1, j].set_title(f"occlusion  C={getattr(r, 'occ_score', np.nan):.2f}", fontsize=7)
        if r.relpath in state["lime"]:
            segs = np.load(seg_cache / (r.relpath.replace("/", "__") + ".npy"))
            wmap = np.clip(state["lime"][r.relpath]["weights"][segs], 0, None)
            axes[2, j].imshow(img)
            axes[2, j].imshow(wmap, cmap="jet", alpha=0.45)
            axes[2, j].set_title(f"LIME  C={getattr(r, 'lime_score', np.nan):.2f}", fontsize=7)
        for i in range(3):
            axes[i, j].set_axis_off()
    fig.suptitle(f"{arm}: leaf mask (yellow), occlusion and LIME positive evidence; "
                 "C = leaf containment", fontsize=9)
    fig.tight_layout()
    fig.savefig(arm_dir / "examples.png", dpi=140)
    plt.close(fig)
    print(f"[{arm}] saved {arm_dir / 'examples.png'}")


# ==========================================================================
# CLI
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="V1: occlusion + LIME + leaf containment")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=[])
    p.add_argument("--summarize", action="store_true",
                   help="only (re)build v1_table.csv from existing summaries")
    p.add_argument("--methods", nargs="+", choices=("occlusion", "lime"),
                   default=["occlusion", "lime"])
    p.add_argument("--ckpt-dir", type=Path, default=None,
                   help="folder with a1_*.pt, a2_*.joblib, a3_cnn.pt, a4_vit.pt "
                        "(default <output-dir>/checkpoints)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=config.RAW_DIR)
    p.add_argument("--splits", type=Path, default=config.SPLIT_FILE)
    p.add_argument("--classes", type=Path, default=None)
    p.add_argument("--mask-root", type=Path, default=None,
                   help="PlantVillage 'segmented' folder (<class>/<name>_final_masked.jpg)")
    p.add_argument("--occ-per-class", type=int, default=10, help="10 x 38 = 380 images")
    p.add_argument("--lime-per-class", type=int, default=3, help="3 x 38 = 114 images")
    p.add_argument("--lime-per-class-a2", type=int, default=2, help="A2 is CPU-bound: 76 images")
    p.add_argument("--patch", type=int, default=32)
    p.add_argument("--fill", choices=("mean", "blur"), default="mean")
    p.add_argument("--lime-samples", type=int, default=200)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU workers for A2 feature extraction")
    p.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    p.add_argument("--fresh", action="store_true", help="ignore saved progress for these arms")
    p.add_argument("--rescore", action="store_true",
                   help="recompute CSV/summary/figure from saved maps without running the model")
    p.add_argument("--check-accuracy", action="store_true",
                   help="only print each arm's clean accuracy on the sample, then stop")
    p.add_argument("--max-leaf-fraction", type=float, default=0.9,
                   help="lifts are undefined above this leaf fraction (default 0.9)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global MAX_LEAF_FRACTION
    MAX_LEAF_FRACTION = args.max_leaf_fraction
    trainer.set_output_root(args.output_dir)
    out_root = config.OUTPUT_DIR / "explain"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.ckpt_dir is None:
        args.ckpt_dir = config.CKPT_DIR

    if args.arms:
        classes = trainer.resolve_classes(args.splits, args.classes)
        sample = select_sample(args.splits, args.occ_per_class, args.lime_per_class, args.seed)
        sample.to_csv(out_root / "sample.csv", index=False)
        mean_rgb = dataset_mean_rgb(args.splits, args.data_root, out_root / "dataset_mean_rgb.json")
        print(f"[explain] sample: {len(sample)} images ({args.occ_per_class}/class), "
              f"LIME on {int(sample['lime'].sum())} (A2: {args.lime_per_class_a2}/class) | "
              f"fill={args.fill} mean RGB={tuple(round(v, 1) for v in mean_rgb)}")
        if args.check_accuracy:
            for arm in args.arms:
                check_accuracy(arm, args, sample, classes)
            return
        masks = MaskIndex(args.mask_root, args.img_size)
        for arm in args.arms:
            run_arm(arm, args, sample, classes, mean_rgb, masks, out_root)

    summarize_all(out_root)


if __name__ == "__main__":
    main()
