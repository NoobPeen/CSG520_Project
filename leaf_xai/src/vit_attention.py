"""
Attention rollout for the ViT arm (Abnar & Zuidema, 2020) -- a supplementary
picture, not a measurement. It never enters the verticals table; occlusion
and LIME in explain.py do that job for every arm.

Why rollout instead of Grad-CAM: a ViT has no final convolutional feature
map, so Grad-CAM has nothing to hook. That gap is part of the argument for
model-agnostic methods (occlusion, LIME) as the shared V1 yardstick.

The idea in one paragraph: each transformer block mixes tokens with an
attention matrix A_l (N x N, N = 1 CLS + 196 patches). Averaging the heads
gives one matrix per block. The residual connection means a token also keeps
a copy of itself, so we add the identity and renormalise each row to sum to
one. Information flow through the whole stack is then the product
R = A_12 * ... * A_1. Row 0 of R (the CLS token) says how much each input
patch contributed to the token the classifier reads. Its 196 patch entries
reshape to 14 x 14 and are upsampled to 224 x 224.

Implementation note: recent timm versions compute attention with a fused
kernel and never materialise the attention matrix, so we cannot just read
it. Instead a forward pre-hook captures each block's attention *input* and
we recompute softmax(q k^T / sqrt(d)) with the block's own qkv weights. This
is exact and works whether or not fused attention is enabled.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class AttentionRollout:
    """
    Usage:
        with AttentionRollout(vit_model) as rollout:
            maps, preds = rollout(images)    # (B, H, W) in [0, 1], (B,)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._inputs: list[torch.Tensor] = []
        self._handles = [blk.attn.register_forward_pre_hook(self._capture)
                         for blk in model.blocks]

    def _capture(self, module, args) -> None:
        self._inputs.append(args[0].detach())

    @staticmethod
    def _attention_matrix(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Recompute the (B, heads, N, N) attention probabilities of one block."""
        B, N, C = x.shape
        H = attn.num_heads
        qkv = attn.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q = attn.q_norm(q) if hasattr(attn, "q_norm") else q
        k = attn.k_norm(k) if hasattr(attn, "k_norm") else k
        scale = getattr(attn, "scale", (C // H) ** -0.5)
        return ((q * scale) @ k.transpose(-2, -1)).softmax(dim=-1)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, head_fusion: str = "mean"):
        self.model.eval()
        self._inputs = []
        logits = self.model(images.float())
        preds = logits.argmax(1)

        rollout = None
        for blk, x in zip(self.model.blocks, self._inputs):
            a = self._attention_matrix(blk.attn, x.float())         # (B, h, N, N)
            a = a.max(dim=1).values if head_fusion == "max" else a.mean(dim=1)
            eye = torch.eye(a.size(-1), device=a.device).unsqueeze(0)
            a = a + eye                                             # residual path
            a = a / a.sum(dim=-1, keepdim=True)                     # rows sum to 1
            rollout = a if rollout is None else a @ rollout

        n_prefix = getattr(self.model, "num_prefix_tokens", 1)
        cls_to_patches = rollout[:, 0, n_prefix:]                   # (B, 196)
        side = int(round(cls_to_patches.size(1) ** 0.5))
        maps = cls_to_patches.reshape(-1, 1, side, side)
        maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear",
                             align_corners=False).squeeze(1)
        flat = maps.flatten(1)
        mn = flat.min(1).values.view(-1, 1, 1)
        mx = flat.max(1).values.view(-1, 1, 1)
        maps = (maps - mn) / (mx - mn + 1e-8)
        return maps.cpu().numpy(), preds.cpu().numpy()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "AttentionRollout":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def _denormalize(t: torch.Tensor, mean, std) -> np.ndarray:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t.detach().cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def save_attention_grid(images: torch.Tensor, maps: np.ndarray, true_labels, pred_labels,
                        out_name: str, mean, std, alpha: float = 0.45) -> Path:
    """Same layout as the Grad-CAM grid so the figures sit side by side."""
    config.ensure_dirs()
    n = min(len(maps), 8)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        photo = _denormalize(images[i], mean, std)
        axes[0, i].imshow(photo)
        axes[0, i].set_axis_off()
        ok = true_labels[i] == pred_labels[i]
        axes[0, i].set_title(f"true: {true_labels[i][:18]}\npred: {pred_labels[i][:18]}",
                             fontsize=7, color="green" if ok else "red")
        axes[1, i].imshow(photo)
        axes[1, i].imshow(maps[i], cmap="jet", alpha=alpha)
        axes[1, i].set_axis_off()
    fig.suptitle("Attention rollout (illustrative): top original, bottom CLS-token attention",
                 fontsize=10)
    fig.tight_layout()
    path = config.FIG_DIR / f"{out_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[vit_attention] saved figure -> {path}")
    return path
