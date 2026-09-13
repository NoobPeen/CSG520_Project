"""
Shared evaluation, plotting and bookkeeping helpers.

Both approaches report the same metrics through the same functions, so the
numbers that end up in the report are computed identically for A1 and A2.
No torch import at module level -- Approach 2 must run without it.
"""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: works on Colab, Kaggle and a plain server
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import config


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = config.SEED) -> None:
    """
    Seed every RNG that can affect a run. torch is seeded only if installed,
    which keeps this module usable from the non-deep-learning approach.

    Note: cudnn.deterministic makes GPU training reproducible but slightly
    slower. For a course project the reproducibility is worth more.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@contextmanager
def timer(label: str):
    """Print how long a block took -- used to report training/inference cost."""
    start = time.perf_counter()
    yield
    print(f"[time] {label}: {time.perf_counter() - start:.1f}s")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Standard multi-class classification metrics.

    Macro-F1 is the headline number rather than accuracy: PlantVillage is
    imbalanced, so plain accuracy is dominated by the largest classes and
    would hide a model that ignores the rare diseases entirely.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_classes": len(classes),
    }

    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] >= 5:
        top5 = np.argsort(-y_prob, axis=1)[:, :5]
        metrics["top5_accuracy"] = float(np.mean([
            y_true[i] in top5[i] for i in range(len(y_true))
        ]))

    metrics["per_class"] = classification_report(
        y_true, y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    return metrics


def save_metrics(metrics: dict, name: str) -> Path:
    """Write a metrics dict to outputs/metrics/<name>.json."""
    config.ensure_dirs()
    path = config.METRIC_DIR / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"[utils] saved metrics -> {path}")
    return path


def print_summary(metrics: dict, title: str) -> None:
    """One compact block per model, easy to copy into the results table."""
    print(f"\n=== {title} ===")
    print(f"  accuracy          : {metrics['accuracy']:.4f}")
    print(f"  balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  macro F1          : {metrics['macro_f1']:.4f}")
    print(f"  weighted F1       : {metrics['weighted_f1']:.4f}")
    if "top5_accuracy" in metrics:
        print(f"  top-5 accuracy    : {metrics['top5_accuracy']:.4f}")
    print(f"  test images       : {metrics['n_samples']}")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_name: str,
    normalize: bool = True,
) -> Path:
    """
    Row-normalised confusion matrix. With ~38 PlantVillage classes the cell
    annotations become unreadable, so they are only drawn for small problems.
    """
    config.ensure_dirs()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, np.where(row_sums == 0, 1, row_sums))
    else:
        cm_display = cm

    n = len(classes)
    size = max(6.0, min(0.35 * n, 18.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm_display, cmap="viridis", vmin=0, vmax=cm_display.max() or 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=max(4, 10 - n // 6))
    ax.set_yticklabels(classes, fontsize=max(4, 10 - n // 6))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({'row-normalised' if normalize else 'counts'})")

    if n <= 15:
        for i in range(n):
            for j in range(n):
                value = cm_display[i, j]
                ax.text(j, i, f"{value:.2f}" if normalize else int(value),
                        ha="center", va="center",
                        color="white" if value < cm_display.max() / 2 else "black",
                        fontsize=7)

    fig.tight_layout()
    path = config.FIG_DIR / f"{out_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[utils] saved figure -> {path}")
    return path


def plot_training_curves(history: dict, out_name: str) -> Path:
    """Loss and macro-F1 per epoch, with the stage boundary marked."""
    config.ensure_dirs()
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_f1"], label="train")
    axes[1].plot(epochs, history["val_f1"], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("macro F1")
    axes[1].set_title("Macro F1"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Mark where the backbone was unfrozen -- the jump there is worth
    # discussing in the report.
    boundary = history.get("warmup_epochs")
    if boundary:
        for ax in axes:
            ax.axvline(boundary + 0.5, color="grey", linestyle="--", linewidth=1)
            ax.text(boundary + 0.6, ax.get_ylim()[1] * 0.95, "unfreeze",
                    fontsize=8, color="grey", va="top")

    fig.tight_layout()
    path = config.FIG_DIR / f"{out_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[utils] saved figure -> {path}")
    return path


def plot_horizontal_bars(
    labels: list[str],
    values: np.ndarray,
    title: str,
    out_name: str,
    xlabel: str = "importance",
    errors: np.ndarray | None = None,
) -> Path:
    """Generic ranked bar chart -- used for the A2 feature importances."""
    config.ensure_dirs()
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = np.asarray(values)[order]
    errs = np.asarray(errors)[order] if errors is not None else None

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.28 * len(labels))))
    ax.barh(range(len(labels)), values, xerr=errs, color="#2e7d32", alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = config.FIG_DIR / f"{out_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[utils] saved figure -> {path}")
    return path
