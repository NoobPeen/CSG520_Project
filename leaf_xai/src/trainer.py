"""
Shared, resumable training loop for Approaches 3 (custom CNN) and 4 (ViT).

A1 keeps its own loop in approach1_transfer.py so the published Phase B
numbers stay reproducible. Everything here follows the same protocol as A1 --
frozen data/splits.csv, the same augmentation policy (torch_data), class-
weighted cross-entropy with label smoothing, mixed precision, early stopping
on validation macro-F1 -- so that when A3/A4 score differently from A1, the
model is the only thing that changed.

What this module adds over A1's loop:

  * Resume after a Colab disconnect. After every epoch the full training
    state (weights, optimiser, scheduler, AMP scaler, history, early-stopping
    counters, RNG state) is written to <tag>_last.pt. Re-running the same
    command picks up at the next epoch. The best-so-far weights live in
    <tag>.pt, written only when validation macro-F1 improves.

  * A list of "stages" instead of hard-coded warm-up/fine-tune. A3 has one
    stage (train everything from scratch); A4 has two (head only, then all).

  * --data-root / --splits / --classes, so the same scripts can later be
    pointed at PlantDoc or Cassava without editing config.py.

  * One prediction interface, ProbaWrapper.predict_proba(uint8 N x H x W x 3)
    -> (N, C) probabilities, which is what explain.py calls for every arm.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms

import config
import data as data_module
import utils
from torch_data import LeafDataset, build_transforms, class_weights_from_dataset


# --------------------------------------------------------------------------
# Output locations
# --------------------------------------------------------------------------
def set_output_root(root: Path | str | None) -> None:
    """
    Redirect every output folder (checkpoints, metrics, figures, preds).

    On Colab the code and images live on the fast local disk (/content) while
    outputs go to Drive, so a disconnect loses nothing. utils.* reads the
    config attributes at call time, so reassigning them here is enough.
    """
    if root is None:
        return
    root = Path(root)
    config.OUTPUT_DIR = root
    config.CKPT_DIR = root / "checkpoints"
    config.FEATURE_DIR = root / "features"
    config.FIG_DIR = root / "figures"
    config.METRIC_DIR = root / "metrics"


def preds_dir() -> Path:
    path = config.OUTPUT_DIR / "preds"
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_save(obj, path: Path) -> None:
    """
    torch.save to a temporary file, then rename over the target. If the
    runtime dies half-way through a write to Drive, the previous checkpoint
    is still intact instead of a truncated file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def resolve_classes(split_file: Path, classes_file: Path | None) -> list[str]:
    """
    Class list, in the frozen index order.

    Priority: an explicit --classes file, then classes.json next to the split
    file (the PlantVillage layout), then the sorted labels in the split.
    """
    split_file = Path(split_file)
    candidates = [classes_file] if classes_file else []
    candidates.append(split_file.parent / "classes.json")
    for cand in candidates:
        if cand is not None and Path(cand).exists():
            return json.loads(Path(cand).read_text())
    df = data_module.load_splits(split_file)
    return sorted(df["label"].unique())


def with_normalization(tfm: transforms.Compose, mean, std) -> transforms.Compose:
    """
    Reuse torch_data's augmentation policy exactly, changing only the
    Normalize constants. ViT-Small was pretrained with mean = std = 0.5, not
    the ImageNet statistics A1's ResNet expects; everything else (crop,
    flips, rotation, colour jitter) stays identical across approaches.
    """
    for t in tfm.transforms:
        if isinstance(t, transforms.Normalize):
            t.mean = list(mean)
            t.std = list(std)
    return tfm


def make_loaders(
    data_root: Path,
    split_file: Path,
    classes: list[str],
    batch_size: int,
    img_size: int,
    num_workers: int,
    max_per_class: int | None,
    mean=config.IMAGENET_MEAN,
    std=config.IMAGENET_STD,
) -> tuple[dict[str, DataLoader], dict[str, LeafDataset]]:
    """Same logic as torch_data.make_dataloaders, but with explicit paths."""
    loaders, datasets = {}, {}
    pin_memory = torch.cuda.is_available()
    for split in ("train", "val", "test"):
        df = data_module.get_split(split, split_file)
        if max_per_class:
            df = data_module.subsample_per_class(df, max_per_class)
        is_train = split == "train"
        tfm = with_normalization(build_transforms(img_size, is_train), mean, std)
        ds = LeafDataset(df, classes, root=data_root, transform=tfm)
        datasets[split] = ds
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=is_train,
            persistent_workers=num_workers > 0,
        )
    return loaders, datasets


def spread_indices(targets: np.ndarray, n: int, seed: int = config.SEED) -> list[int]:
    """
    Pick n example images from n different classes (fixed seed). The test
    split is sorted by label, so 'the first batch' would be eight images of
    the same disease -- useless for a figure.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    chosen_classes = rng.choice(classes, size=min(n, len(classes)), replace=False)
    return [int(rng.choice(np.flatnonzero(targets == c))) for c in sorted(chosen_classes)]


def load_eval_image_uint8(path: Path | str, img_size: int = config.IMG_SIZE) -> np.ndarray:
    """
    The exact geometric preprocessing used at evaluation time (resize to
    1.14 x size, centre-crop), returned as uint8 HxWx3 *before* normalisation.

    explain.py must build its images (and crop its leaf masks) with this, so
    that the pixels it perturbs are the pixels the model was scored on.
    """
    geo = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
    ])
    return np.array(geo(Image.open(path).convert("RGB")), dtype=np.uint8)  # writable copy


# --------------------------------------------------------------------------
# Optimiser helpers
# --------------------------------------------------------------------------
def param_groups(params: Iterable[tuple[str, nn.Parameter]], weight_decay: float):
    """
    AdamW groups: no weight decay on biases, norm scales, the ViT position
    embedding or CLS token (all 1-D or 'embedding-like'). Decaying those
    shrinks them towards zero for no regularisation benefit.
    """
    decay, no_decay = [], []
    for name, p in params:
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(("pos_embed", "cls_token", "reg_token")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@dataclass
class Stage:
    """
    One phase of training.

    setup(model)       -- freeze / unfreeze whatever this stage needs
    lr, weight_decay   -- AdamW settings for the stage
    cosine             -- cosine-decay the LR over the stage's epochs
    early_stop         -- whether patience is checked in this stage
    train_mode(model)  -- optional hook called after model.train() each epoch
                          (e.g. keep frozen BatchNorm layers in eval mode)
    """
    name: str
    epochs: int
    lr: float
    weight_decay: float
    setup: Callable[[nn.Module], None]
    cosine: bool = True
    early_stop: bool = True
    min_lr_ratio: float = 0.01
    train_mode: Callable[[nn.Module], None] | None = None


# --------------------------------------------------------------------------
# One epoch of training / evaluation
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_amp: bool, train_mode=None) -> tuple[float, float]:
    model.train()
    if train_mode is not None:
        train_mode(model)

    running_loss, seen = 0.0, 0
    preds, targets_all = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)          # clip the real gradients, not the scaled ones
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        seen += images.size(0)
        preds.append(logits.argmax(1).detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

    y_pred, y_true = np.concatenate(preds), np.concatenate(targets_all)
    return running_loss / max(seen, 1), f1_score(y_true, y_pred, average="macro",
                                                 zero_division=0)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool):
    """Returns (loss, macro_f1, y_true, y_pred, y_prob)."""
    model.eval()
    running_loss, seen = 0.0, 0
    probs, preds, targets_all = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)
        seen += images.size(0)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        targets_all.append(targets.cpu().numpy())
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets_all)
    return (running_loss / max(seen, 1),
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            y_true, y_pred, y_prob)


# --------------------------------------------------------------------------
# Checkpoint formats
# --------------------------------------------------------------------------
def best_checkpoint(model, classes, meta: dict, epoch: int, val_f1: float) -> dict:
    """
    What <tag>.pt holds: weights plus everything needed to use them without
    the training script -- class order, normalisation, input size, and the
    model-construction arguments.
    """
    return {
        "state_dict": model.state_dict(),
        "classes": list(classes),
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "epoch": epoch,
        "val_macro_f1": val_f1,
        **meta,
    }


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except RuntimeError:
            pass    # different GPU count after reconnect; not worth failing over


# --------------------------------------------------------------------------
# The resumable multi-stage loop
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    history: dict
    best_val_f1: float
    best_epoch: int
    resumed_from: int = 0
    notes: list[str] = field(default_factory=list)


def fit(
    model: nn.Module,
    stages: list[Stage],
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    tag: str,
    classes: list[str],
    meta: dict,
    patience: int,
    fresh: bool = False,
    log_prefix: str | None = None,
) -> FitResult:
    """
    Train through `stages` in order with early stopping on val macro-F1.

    Files:
      <CKPT_DIR>/<tag>.pt       best weights (+ class mapping, meta)
      <CKPT_DIR>/<tag>_last.pt  full resume state, rewritten every epoch
    Pass fresh=True to ignore an existing _last.pt and start over.
    """
    p = log_prefix or f"[{tag}]"
    config.ensure_dirs()
    best_path = config.CKPT_DIR / f"{tag}.pt"
    last_path = config.CKPT_DIR / f"{tag}_last.pt"

    total_epochs = sum(s.epochs for s in stages)
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [],
               "lr": [], "stage": [], "epoch_seconds": []}
    if len(stages) > 1:
        history["warmup_epochs"] = stages[0].epochs   # utils draws the 'unfreeze' line here
    best_f1, best_epoch, no_gain = -1.0, -1, 0
    done_epochs = 0
    resume = None

    # ---------------- resume -------------------------------------------------
    if last_path.exists() and not fresh:
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume.get("stage_plan") != [(s.name, s.epochs) for s in stages]:
            print(f"{p} WARNING: {last_path.name} was written with stage plan "
                  f"{resume.get('stage_plan')}, now {[(s.name, s.epochs) for s in stages]}. "
                  "Resuming anyway; pass --fresh to start over.")
        model.load_state_dict(resume["model"])
        history = resume["history"]
        best_f1, best_epoch = resume["best_f1"], resume["best_epoch"]
        no_gain, done_epochs = resume["no_gain"], resume["epoch"]
        _set_rng_state(resume["rng"])
        if resume.get("finished"):
            print(f"{p} training already finished at epoch {done_epochs} "
                  f"(best val macro-F1 {best_f1:.4f} @ epoch {best_epoch}); skipping to evaluation")
            history["best_val_f1"], history["best_epoch"] = best_f1, best_epoch
            return FitResult(history, best_f1, best_epoch, resumed_from=done_epochs)
        print(f"{p} resuming after epoch {done_epochs}/{total_epochs} "
              f"(best val macro-F1 so far {best_f1:.4f} @ epoch {best_epoch})")

    scaler = make_scaler(device, use_amp)
    if resume is not None and resume.get("scaler"):
        scaler.load_state_dict(resume["scaler"])

    global_epoch = 0
    stopped = False
    for si, stage in enumerate(stages):
        stage_start = global_epoch
        stage_end = global_epoch + stage.epochs
        if done_epochs >= stage_end:           # stage fully done before the disconnect
            global_epoch = stage_end
            continue

        stage.setup(model)
        optimizer = torch.optim.AdamW(
            param_groups(model.named_parameters(), stage.weight_decay), lr=stage.lr
        )
        scheduler = None
        if stage.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage.epochs, eta_min=stage.lr * stage.min_lr_ratio
            )

        # Mid-stage resume: restore optimiser + scheduler only if they belong
        # to this stage (a new stage always starts with fresh moments).
        if resume is not None and resume.get("stage_index") == si and done_epochs > stage_start:
            optimizer.load_state_dict(resume["optimizer"])
            if scheduler is not None and resume.get("scheduler"):
                scheduler.load_state_dict(resume["scheduler"])
        n_trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        print(f"{p} stage '{stage.name}': epochs {stage_start + 1}-{stage_end}, "
              f"lr {stage.lr:g}, wd {stage.weight_decay:g}, trainable params {n_trainable:,}")

        for epoch in range(max(done_epochs, stage_start) + 1, stage_end + 1):
            started = time.perf_counter()
            tr_loss, tr_f1 = train_one_epoch(model, loaders["train"], criterion, optimizer,
                                             scaler, device, use_amp, stage.train_mode)
            va_loss, va_f1, *_ = evaluate(model, loaders["val"], criterion, device, use_amp)
            lr_now = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            secs = time.perf_counter() - started

            for k, v in (("train_loss", tr_loss), ("val_loss", va_loss), ("train_f1", tr_f1),
                         ("val_f1", va_f1), ("lr", lr_now), ("stage", stage.name),
                         ("epoch_seconds", secs)):
                history[k].append(v)
            print(f"{p} epoch {epoch:02d}/{total_epochs} ({stage.name}) "
                  f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | "
                  f"val_loss={va_loss:.4f} val_f1={va_f1:.4f} ({secs:.0f}s)")

            # Select on macro-F1, not loss: the loss is class-weighted and
            # label-smoothed, so it is not the metric we report.
            if va_f1 > best_f1:
                best_f1, best_epoch, no_gain = va_f1, epoch, 0
                atomic_save(best_checkpoint(model, classes, meta, epoch, va_f1), best_path)
                print(f"{p}   new best val macro-F1 {best_f1:.4f} -> {best_path.name}")
            else:
                no_gain += 1

            stopped = stage.early_stop and no_gain >= patience
            finished = stopped or (epoch == total_epochs)
            atomic_save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "history": history,
                "best_f1": best_f1, "best_epoch": best_epoch, "no_gain": no_gain,
                "epoch": epoch, "stage_index": si,
                "stage_plan": [(s.name, s.epochs) for s in stages],
                "finished": finished,
                "rng": _rng_state(),
                "meta": meta,
            }, last_path)

            if stopped:
                print(f"{p} early stop: no val macro-F1 gain for {patience} epochs")
                break
        global_epoch = stage_end
        if stopped:
            break

    history["best_val_f1"] = best_f1
    history["best_epoch"] = best_epoch
    (config.METRIC_DIR / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    return FitResult(history, best_f1, best_epoch,
                     resumed_from=done_epochs if resume is not None else 0)


# --------------------------------------------------------------------------
# Final evaluation + the files every approach must leave behind
# --------------------------------------------------------------------------
def load_best(model: nn.Module, tag: str, classes: list[str], device) -> dict:
    path = config.CKPT_DIR / f"{tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}; train first.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if list(ckpt["classes"]) != list(classes):
        raise ValueError(f"{path.name} was trained on a different class list/order "
                         "than the current split. Refusing to score it.")
    model.load_state_dict(ckpt["state_dict"])
    return ckpt


def final_report(
    model, loaders, datasets, criterion, device, use_amp,
    tag: str, classes: list[str], extra: dict, history: dict | None,
    title: str,
) -> dict:
    """
    Score the best checkpoint on the test split and write:
      metrics/<tag>.json, preds/<tag>_test_probs.npz,
      figures/<tag>_confusion.png, figures/<tag>_curves.png
    """
    test_loss, _, y_true, y_pred, y_prob = evaluate(model, loaders["test"], criterion,
                                                    device, use_amp)
    metrics = utils.compute_metrics(y_true, y_pred, classes, y_prob)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), average="macro", zero_division=0)
    metrics.update({"macro_precision": float(prec), "macro_recall": float(rec),
                    "test_loss": float(test_loss), **extra})
    if history:
        metrics["history"] = history

    utils.print_summary(metrics, title)
    print(f"  macro precision   : {prec:.4f}\n  macro recall      : {rec:.4f}")
    utils.save_metrics(metrics, tag)
    utils.plot_confusion_matrix(y_true, y_pred, classes, f"{tag}_confusion")

    # History from this run, or -- for --eval-only -- from the saved JSON,
    # so the curves figure is never silently missing.
    hist_file = config.METRIC_DIR / f"{tag}_history.json"
    if not history and hist_file.exists():
        history = json.loads(hist_file.read_text())
    if history and history.get("train_loss"):
        utils.plot_training_curves(history, f"{tag}_curves")

    # Full probability matrix: V3 (confidence) is computed from this alone.
    npz = preds_dir() / f"{tag}_test_probs.npz"
    np.savez_compressed(
        npz, probs=y_prob.astype(np.float32), y_true=y_true, y_pred=y_pred,
        relpaths=np.array(datasets["test"].df["relpath"].tolist()),
        classes=np.array(classes),
    )
    print(f"[{tag}] saved test probabilities -> {npz}")
    return metrics


# --------------------------------------------------------------------------
# The single interface explain.py uses for every approach
# --------------------------------------------------------------------------
class ProbaWrapper:
    """
    predict_proba(images) with images a uint8 array N x H x W x 3 (RGB,
    already resized/cropped to the model's input size, e.g. with
    load_eval_image_uint8) -> float32 array N x C of softmax probabilities.

    Normalisation is the model's own (ImageNet stats for A3, 0.5/0.5 for the
    ViT), so explain.py never needs to know which arm it is perturbing.
    """

    def __init__(self, model: nn.Module, mean, std, classes: list[str],
                 device: torch.device | str | None = None, batch_size: int = 64,
                 use_amp: bool = True) -> None:
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.classes = list(classes)
        self.batch_size = batch_size
        self.use_amp = use_amp and self.device.type == "cuda"

    @torch.no_grad()
    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"expected N x H x W x 3 uint8, got shape {images.shape}")
        out = []
        for start in range(0, len(images), self.batch_size):
            chunk = torch.from_numpy(np.ascontiguousarray(images[start:start + self.batch_size]))
            x = chunk.to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - self.mean) / self.std
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    __call__ = predict_proba


# --------------------------------------------------------------------------
# CLI pieces shared by both scripts
# --------------------------------------------------------------------------
def add_common_args(parser, *, epochs: int, batch_size: int, lr: float,
                    patience: int, weight_decay: float) -> None:
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight-decay", type=float, default=weight_decay)
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="cap images per class in every split (0 = use everything)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--data-root", type=Path, default=config.RAW_DIR,
                        help="folder holding <class>/<image> (default data/raw)")
    parser.add_argument("--splits", type=Path, default=config.SPLIT_FILE,
                        help="frozen split CSV (default data/splits.csv)")
    parser.add_argument("--classes", type=Path, default=None,
                        help="class-order JSON (default: classes.json beside --splits)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where checkpoints/metrics/figures/preds go (default outputs/). "
                             "On Colab point this at Drive.")
    parser.add_argument("--tag", type=str, default=None,
                        help="override the output tag (e.g. for a PlantDoc run)")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore <tag>_last.pt and train from epoch 1")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training and score the saved best checkpoint")
