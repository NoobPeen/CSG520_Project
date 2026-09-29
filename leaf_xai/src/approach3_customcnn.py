"""
APPROACH 3 -- Custom CNN trained from scratch.

A plain VGG-style network (models/custom_cnn.py, ~2.4M parameters, no
residual connections, random initialisation). It answers a different
question from A1: what does a network learn about *where to look* when it has
no ImageNet prior at all? Unlike A1's --no-pretrained ablation, which was cut
short, this is a full scratch run with a schedule designed for scratch
training.

Training protocol (identical to A1 wherever the model does not force a
difference):
  * frozen data/splits.csv, same augmentation policy (torch_data)
  * class-weighted cross-entropy, label smoothing 0.05, mixed precision
  * early stopping on validation macro-F1 (patience 6 here)
  * AdamW, lr 1e-3, cosine decay, single stage. There is no warm-up stage
    because there is no pretrained backbone to protect from a random head.

Outputs (tag a3_cnn):
  outputs/checkpoints/a3_cnn.pt          best weights + class mapping
  outputs/checkpoints/a3_cnn_last.pt     resume state (every epoch)
  outputs/metrics/a3_cnn.json            accuracy, P/R, macro-F1, per class
  outputs/preds/a3_cnn_test_probs.npz    full test probability matrix
  outputs/figures/a3_cnn_confusion.png, a3_cnn_curves.png, a3_cnn_gradcam.png

Grad-CAM here is an illustrative picture only; the V1 measurement
(occlusion + LIME + leaf containment) is done by explain.py for every arm.

    python src/approach3_customcnn.py                    # train (resumes if interrupted)
    python src/approach3_customcnn.py --eval-only        # re-score saved checkpoint
    python src/approach3_customcnn.py --max-per-class 30 --epochs 2   # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import config
import trainer
import utils
from gradcam import GradCAM, save_cam_grid
from models.custom_cnn import CustomCNN, count_parameters
from torch_data import class_weights_from_dataset

TAG = "a3_cnn"
MEAN, STD = config.IMAGENET_MEAN, config.IMAGENET_STD   # plain standardisation; no pretrained stats to match


def build_model(num_classes: int, dropout: float = 0.3) -> CustomCNN:
    return CustomCNN(num_classes=num_classes, dropout=dropout)


def run_gradcam(model, datasets, classes, device, tag: str, n_images: int = 8) -> None:
    """Grad-CAM on 8 test images from 8 different classes (fixed seed)."""
    ds = datasets["test"]
    idx = trainer.spread_indices(ds.targets, n_images)
    images = torch.stack([ds[i][0] for i in idx]).to(device)
    targets = [int(ds.targets[i]) for i in idx]

    with GradCAM(model, model.gradcam_layer) as cam_extractor:
        cams, pred_idx = cam_extractor(images)

    save_cam_grid(
        images=images.cpu(), cams=cams,
        true_labels=[classes[i] for i in targets],
        pred_labels=[classes[i] for i in pred_idx],
        out_name=f"{tag}_gradcam",
    )
    print(f"[{tag}] NOTE: the Grad-CAM figure is illustrative only -- "
          "it does not enter the verticals table.")


def run(args: argparse.Namespace) -> dict:
    trainer.set_output_root(args.output_dir)
    utils.set_seed(args.seed)
    config.ensure_dirs()
    tag = args.tag or TAG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"[{tag}] device={device} amp={use_amp} output={config.OUTPUT_DIR}")

    classes = trainer.resolve_classes(args.splits, args.classes)
    loaders, datasets = trainer.make_loaders(
        args.data_root, args.splits, classes, args.batch_size, args.img_size,
        args.workers, args.max_per_class, MEAN, STD,
    )
    print(f"[{tag}] {len(classes)} classes | train={len(datasets['train'])} "
          f"val={len(datasets['val'])} test={len(datasets['test'])}")

    model = build_model(len(classes), args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"[{tag}] CustomCNN parameters: {n_params:,}")

    weights = class_weights_from_dataset(datasets["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.LABEL_SMOOTHING)

    meta = {"approach": "A3_custom_cnn", "arch": "custom_cnn_vgg5",
            "dropout": args.dropout, "img_size": args.img_size,
            "mean": list(MEAN), "std": list(STD), "n_params": n_params}

    history = None
    if not args.eval_only:
        stages = [trainer.Stage(
            name="scratch", epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay,
            setup=lambda m: [p.requires_grad_(True) for p in m.parameters()],
            cosine=True, early_stop=True,
        )]
        result = trainer.fit(model, stages, loaders, criterion, device, use_amp,
                             tag, classes, meta, patience=args.patience, fresh=args.fresh)
        history = result.history

    trainer.load_best(model, tag, classes, device)
    metrics = trainer.final_report(
        model, loaders, datasets, criterion, device, use_amp, tag, classes,
        extra={**meta, "batch_size": args.batch_size, "lr": args.lr,
               "weight_decay": args.weight_decay, "max_epochs": args.epochs,
               "patience": args.patience, "max_per_class": args.max_per_class,
               "data_root": str(args.data_root), "splits": str(args.splits)},
        history=history, title=f"A3 / custom CNN / test set",
    )

    if args.gradcam:
        run_gradcam(model, datasets, classes, device, tag)
    return metrics


# --------------------------------------------------------------------------
# explain.py entry point
# --------------------------------------------------------------------------
def load_predictor(ckpt_path: Path | str | None = None, device=None,
                   batch_size: int = 128) -> trainer.ProbaWrapper:
    """
    Rebuild A3 from its checkpoint and return the shared predict_proba
    wrapper:  predictor.predict_proba(uint8 N x 224 x 224 x 3) -> (N, 38).
    """
    ckpt_path = Path(ckpt_path) if ckpt_path else config.CKPT_DIR / f"{TAG}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(len(ckpt["classes"]), ckpt.get("dropout", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    return trainer.ProbaWrapper(model, ckpt["mean"], ckpt["std"], ckpt["classes"],
                                device=device, batch_size=batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approach 3: custom CNN from scratch")
    trainer.add_common_args(parser, epochs=30, batch_size=64, lr=1e-3,
                            patience=6, weight_decay=config.WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gradcam", action=argparse.BooleanOptionalAction, default=True,
                        help="save the illustrative Grad-CAM grid (default on)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
