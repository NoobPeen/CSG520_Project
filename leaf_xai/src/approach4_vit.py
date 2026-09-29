"""
APPROACH 4 -- Vision Transformer (ViT-Small/16), ImageNet-pretrained.

timm's vit_small_patch16_224 with the 1000-way head replaced by one for our
classes (~22M parameters). The image is cut into 196 patches of 16x16, each
embedded to 384 dimensions; a CLS token and position embeddings are added;
12 transformer blocks (6-head self-attention + MLP) mix the tokens; the final
LayerNorm'd CLS token goes through Linear(384 -> classes).

Why it is in the project: a CNN's receptive field grows gradually, so its
evidence is built from local texture first. Self-attention lets any patch
attend to any other patch from layer 1, including background patches. Whether
that makes the ViT lean on the background more or less than a CNN is exactly
what the leaf-containment metric (explain.py) is there to measure.

Training uses A1's two-stage schedule so the comparison with A1 is clean:
  Stage 1 (3 epochs)  backbone frozen, head only, lr 1e-3
  Stage 2 (~10 epochs) everything unfrozen, lr 3e-5 (ViTs are more fragile
                      than CNNs under fine-tuning), cosine decay, weight
                      decay 0.05, early stopping on val macro-F1, keep best.
No layer-wise LR decay -- left out on purpose to keep the recipe simple.
The images are normalised with the ViT's own pretraining statistics
(mean = std = 0.5), taken from timm's pretrained config; the augmentation
policy is otherwise identical to A1.

No Grad-CAM: a ViT has no final conv map. Attention rollout
(vit_attention.py) gives an illustrative picture instead.

Outputs (tag a4_vit): same set as A3, plus figures/a4_vit_attention.png.

    python src/approach4_vit.py                      # train (resumes if interrupted)
    python src/approach4_vit.py --eval-only
    python src/approach4_vit.py --max-per-class 30 --warmup-epochs 1 --epochs 1   # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import config
import trainer
import utils
from torch_data import class_weights_from_dataset
from vit_attention import AttentionRollout, save_attention_grid

TAG = "a4_vit"
MODEL_NAME = "vit_small_patch16_224"


def _timm():
    try:
        import timm
    except ImportError as exc:
        raise ImportError("A4 needs timm:  pip install timm") from exc
    return timm


def build_model(num_classes: int, pretrained: bool = True, drop_path: float = 0.0,
                model_name: str = MODEL_NAME) -> nn.Module:
    timm = _timm()
    return timm.create_model(model_name, pretrained=pretrained,
                             num_classes=num_classes, drop_path_rate=drop_path)


def norm_stats(model: nn.Module) -> tuple[tuple, tuple]:
    """The mean/std the checkpoint was pretrained with (0.5/0.5 for augreg ViTs)."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return tuple(cfg.get("mean", (0.5, 0.5, 0.5))), tuple(cfg.get("std", (0.5, 0.5, 0.5)))


def head_only(model: nn.Module) -> None:
    head = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def run_attention(model, datasets, classes, device, tag, mean, std, n_images: int = 8) -> None:
    ds = datasets["test"]
    idx = trainer.spread_indices(ds.targets, n_images)
    images = torch.stack([ds[i][0] for i in idx]).to(device)
    targets = [int(ds.targets[i]) for i in idx]
    with AttentionRollout(model) as rollout:
        maps, preds = rollout(images)
    save_attention_grid(images.cpu(), maps, [classes[i] for i in targets],
                        [classes[i] for i in preds], f"{tag}_attention", mean, std)
    print(f"[{tag}] NOTE: attention rollout is illustrative only -- "
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
    # In eval-only mode the checkpoint supplies the weights, so skip the download.
    pretrained = not args.no_pretrained and not args.eval_only
    model = build_model(len(classes), pretrained=pretrained, drop_path=args.drop_path).to(device)
    mean, std = norm_stats(model)
    best_path = config.CKPT_DIR / f"{tag}.pt"
    if args.eval_only and best_path.exists():   # score with the stats it was trained with
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        mean, std = tuple(saved["mean"]), tuple(saved["std"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] {MODEL_NAME} pretrained={pretrained} params={n_params:,} "
          f"norm mean={mean} std={std}")

    loaders, datasets = trainer.make_loaders(
        args.data_root, args.splits, classes, args.batch_size, args.img_size,
        args.workers, args.max_per_class, mean, std,
    )
    print(f"[{tag}] {len(classes)} classes | train={len(datasets['train'])} "
          f"val={len(datasets['val'])} test={len(datasets['test'])}")

    weights = class_weights_from_dataset(datasets["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.LABEL_SMOOTHING)

    meta = {"approach": "A4_vit", "arch": MODEL_NAME, "pretrained": not args.no_pretrained,
            "drop_path": args.drop_path, "img_size": args.img_size,
            "mean": list(mean), "std": list(std), "n_params": n_params}

    history = None
    if not args.eval_only:
        stages = []
        if args.warmup_epochs > 0:
            stages.append(trainer.Stage(
                name="warmup", epochs=args.warmup_epochs, lr=args.head_lr,
                weight_decay=args.weight_decay, setup=head_only,
                cosine=False, early_stop=False,
            ))
        stages.append(trainer.Stage(
            name="finetune", epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, setup=unfreeze_all,
            cosine=True, early_stop=True,
        ))
        result = trainer.fit(model, stages, loaders, criterion, device, use_amp,
                             tag, classes, meta, patience=args.patience, fresh=args.fresh)
        history = result.history

    trainer.load_best(model, tag, classes, device)
    metrics = trainer.final_report(
        model, loaders, datasets, criterion, device, use_amp, tag, classes,
        extra={**meta, "batch_size": args.batch_size, "head_lr": args.head_lr,
               "lr": args.lr, "weight_decay": args.weight_decay,
               "warmup_epochs": args.warmup_epochs, "finetune_epochs": args.epochs,
               "patience": args.patience, "max_per_class": args.max_per_class,
               "data_root": str(args.data_root), "splits": str(args.splits)},
        history=history, title="A4 / ViT-Small / test set",
    )

    if args.attention:
        run_attention(model, datasets, classes, device, tag, mean, std)
    return metrics


# --------------------------------------------------------------------------
# explain.py entry point
# --------------------------------------------------------------------------
def load_predictor(ckpt_path: Path | str | None = None, device=None,
                   batch_size: int = 64) -> trainer.ProbaWrapper:
    """
    Rebuild A4 from its checkpoint (no weight download) and return the shared
    wrapper:  predictor.predict_proba(uint8 N x 224 x 224 x 3) -> (N, 38).
    """
    ckpt_path = Path(ckpt_path) if ckpt_path else config.CKPT_DIR / f"{TAG}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(len(ckpt["classes"]), pretrained=False,
                        model_name=ckpt.get("arch", MODEL_NAME))
    model.load_state_dict(ckpt["state_dict"])
    return trainer.ProbaWrapper(model, ckpt["mean"], ckpt["std"], ckpt["classes"],
                                device=device, batch_size=batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approach 4: ViT-Small fine-tuning")
    trainer.add_common_args(parser, epochs=10, batch_size=64, lr=3e-5,
                            patience=config.EARLY_STOP_PATIENCE, weight_decay=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=config.WARMUP_EPOCHS,
                        help="stage 1: head-only epochs (default 3)")
    parser.add_argument("--head-lr", type=float, default=config.HEAD_LR,
                        help="stage 1 learning rate (default 1e-3)")
    parser.add_argument("--drop-path", type=float, default=0.0,
                        help="stochastic depth rate (off by default)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random init (for tests / an ablation; not the main run)")
    parser.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True,
                        help="save the illustrative attention-rollout grid (default on)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
