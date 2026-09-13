"""
APPROACH 1 -- Transfer learning.

ResNet50 pretrained on ImageNet, fine-tuned on the leaf dataset, with
EfficientNet-B0 as a second backbone to check that the findings are not an
artefact of one architecture.

Training is done in two stages, which is the part worth explaining in the
report:

  Stage 1 (warm-up). The backbone is frozen and only the freshly initialised
  classifier head is trained. A random head produces large, meaningless
  gradients; letting those flow into pretrained weights in the first few
  hundred steps destroys the ImageNet features we came here for.

  Stage 2 (fine-tune). Everything is unfrozen and trained at a much smaller
  learning rate with cosine decay, so the pretrained filters are nudged
  towards leaf textures rather than overwritten.

Batch-norm layers in the frozen backbone are kept in eval mode during stage 1,
otherwise their running statistics would drift towards the new data while the
weights that depend on them are still frozen.

Typical run:

    python src/data.py --build
    python src/approach1_transfer.py --arch resnet50
    python src/approach1_transfer.py --arch efficientnet_b0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision import models

import config
import utils
from gradcam import GradCAM, get_target_layer, save_cam_grid
from torch_data import class_weights_from_dataset, make_dataloaders

ARCHITECTURES = ("resnet50", "efficientnet_b0")


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
def build_model(arch: str, num_classes: int, dropout: float = 0.2,
                pretrained: bool = True) -> nn.Module:
    """
    Load an ImageNet backbone and replace its 1000-way classifier with a
    head sized for our classes. Only the head is new; everything below it
    arrives already knowing edges, textures and colour blobs.
    """
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"arch must be one of {ARCHITECTURES}, got '{arch}'")

    return model


def get_head(model: nn.Module, arch: str) -> nn.Module:
    """The newly initialised classifier, i.e. the only part trained in stage 1."""
    return model.fc if arch.startswith("resnet") else model.classifier


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    """Freeze or unfreeze everything except the classifier head."""
    head = get_head(model, arch)
    head_params = {id(p) for p in head.parameters()}
    for param in model.parameters():
        if id(param) not in head_params:
            param.requires_grad = trainable


def freeze_backbone_bn(model: nn.Module, arch: str) -> None:
    """
    Put every BatchNorm layer outside the head into eval mode, so its running
    mean/variance stay at the ImageNet values while the backbone is frozen.
    """
    head = get_head(model, arch)
    head_modules = set(id(m) for m in head.modules())
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and id(module) not in head_modules:
            module.eval()


# --------------------------------------------------------------------------
# Mixed-precision helpers (kept version-tolerant across torch releases)
# --------------------------------------------------------------------------
def make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------
# Train / evaluate loops
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_amp: bool, freeze_bn: bool, arch: str) -> tuple[float, float]:
    model.train()
    if freeze_bn:
        freeze_backbone_bn(model, arch)

    running_loss, seen = 0.0, 0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        # Unscale before clipping, otherwise the clip threshold applies to the
        # scaled gradients and does nothing useful.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        seen += images.size(0)
        all_preds.append(logits.argmax(1).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
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
        prob = torch.softmax(logits.float(), dim=1)
        probs.append(prob.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        targets_all.append(targets.cpu().numpy())

    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets_all)
    loss = running_loss / max(seen, 1)
    return loss, f1_score(y_true, y_pred, average="macro", zero_division=0), \
        y_true, y_pred, y_prob


# --------------------------------------------------------------------------
# Grad-CAM on a handful of test images
# --------------------------------------------------------------------------
def run_gradcam(model, loaders, classes, arch: str, device, n_images: int = 8) -> None:
    images, targets = next(iter(loaders["test"]))
    images = images[:n_images].to(device)
    targets = targets[:n_images]

    with GradCAM(model, get_target_layer(model, arch)) as cam_extractor:
        cams, pred_idx = cam_extractor(images)

    save_cam_grid(
        images=images,
        cams=cams,
        true_labels=[classes[i] for i in targets.numpy()],
        pred_labels=[classes[i] for i in pred_idx],
        out_name=f"a1_{arch}_gradcam",
    )


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    utils.set_seed(args.seed)
    config.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"[a1] device={device}  amp={use_amp}  arch={args.arch}")

    loaders, classes, datasets = make_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.workers,
        max_per_class=args.max_per_class,
        balanced_sampler=args.balanced_sampler,
    )
    print(f"[a1] {len(classes)} classes | "
          f"train={len(datasets['train'])} val={len(datasets['val'])} "
          f"test={len(datasets['test'])}")

    model = build_model(args.arch, num_classes=len(classes),
                        dropout=args.dropout,
                        pretrained=not args.no_pretrained).to(device)

    # Class-weighted loss counters the imbalance; label smoothing stops the
    # network becoming over-confident on the visually easy classes.
    weights = class_weights_from_dataset(datasets["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=config.LABEL_SMOOTHING)

    ckpt_path = config.CKPT_DIR / f"a1_{args.arch}.pt"

    if args.eval_only:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}; train first.")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        history = {}
    else:
        history = train(model, loaders, criterion, device, use_amp, args, ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ---- final evaluation on the held-out test split -----------------------
    test_loss, test_f1, y_true, y_pred, y_prob = evaluate(
        model, loaders["test"], criterion, device, use_amp
    )
    metrics = utils.compute_metrics(y_true, y_pred, classes, y_prob)
    metrics.update({
        "approach": "A1_transfer_learning",
        "architecture": args.arch,
        "pretrained": not args.no_pretrained,
        "test_loss": test_loss,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs,
        "finetune_epochs": args.epochs,
        "history": history,
    })

    utils.print_summary(metrics, f"A1 / {args.arch} / test set")
    utils.save_metrics(metrics, f"a1_{args.arch}")
    utils.plot_confusion_matrix(y_true, y_pred, classes, f"a1_{args.arch}_confusion")
    if history:
        utils.plot_training_curves(history, f"a1_{args.arch}_curves")

    if not args.no_gradcam:
        run_gradcam(model, loaders, classes, args.arch, device)

    return metrics


def train(model, loaders, criterion, device, use_amp, args, ckpt_path: Path) -> dict:
    """Two-stage schedule with early stopping on validation macro-F1."""
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [],
               "lr": [], "warmup_epochs": args.warmup_epochs}
    scaler = make_scaler(device, use_amp)
    best_f1, best_epoch, epochs_without_gain = -1.0, -1, 0

    # ---------------- Stage 1: head only ----------------------------------
    set_backbone_trainable(model, args.arch, trainable=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.head_lr, weight_decay=config.WEIGHT_DECAY,
    )

    total_epochs = args.warmup_epochs + args.epochs
    scheduler = None

    for epoch in range(1, total_epochs + 1):
        stage = "warmup" if epoch <= args.warmup_epochs else "finetune"

        # ---------------- Stage 2 begins: unfreeze everything -------------
        if epoch == args.warmup_epochs + 1:
            print("[a1] unfreezing backbone, switching to fine-tune LR")
            set_backbone_trainable(model, args.arch, trainable=True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )

        started = time.perf_counter()
        train_loss, train_f1 = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device,
            use_amp, freeze_bn=(stage == "warmup"), arch=args.arch,
        )
        val_loss, val_f1, *_ = evaluate(model, loaders["val"], criterion,
                                        device, use_amp)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"[a1] epoch {epoch:02d}/{total_epochs} ({stage}) "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
              f"({time.perf_counter() - started:.0f}s)")

        # Model selection on macro-F1, not on loss: the loss is class-weighted
        # and label-smoothed, so it is not directly comparable to the metric
        # we actually report.
        if val_f1 > best_f1:
            best_f1, best_epoch, epochs_without_gain = val_f1, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"[a1]   new best val macro-F1 {best_f1:.4f} -> saved {ckpt_path.name}")
        else:
            epochs_without_gain += 1
            if stage == "finetune" and epochs_without_gain >= args.patience:
                print(f"[a1] early stop: no improvement for {args.patience} epochs")
                break

    history["best_val_f1"] = best_f1
    history["best_epoch"] = best_epoch
    (config.METRIC_DIR / f"a1_{args.arch}_history.json").write_text(
        json.dumps(history, indent=2)
    )
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approach 1: transfer learning")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="resnet50")
    parser.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS,
                        help="fine-tuning epochs (after warm-up)")
    parser.add_argument("--warmup-epochs", type=int, default=config.WARMUP_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--lr", type=float, default=config.FINETUNE_LR)
    parser.add_argument("--head-lr", type=float, default=config.HEAD_LR)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="cap images per class (0 = use everything)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random initialisation instead of ImageNet weights. "
                             "This is the ablation that answers A1's question: "
                             "does starting from general photographs actually "
                             "help the model find the leaf?")
    parser.add_argument("--no-gradcam", action="store_true")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training and evaluate the saved checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
