"""
Grad-CAM for the CNN approaches, written from the definition rather than
pulled from a library, so the mechanics are auditable in the report.

Idea, in one paragraph: the last convolutional block outputs K feature maps
A^k of size h x w. Each map responds to some visual pattern, and its spatial
layout says *where* that pattern occurred. To find how much map k mattered for
class c, take the gradient of the class score y^c with respect to that map and
average it over space -- that global-average-pooled gradient is the map's
weight, alpha_k. The class-discriminative heatmap is then the weighted sum of
the maps followed by a ReLU, which keeps only the evidence that pushes the
score up rather than down. Upsampling that h x w map back to the input size
gives the familiar overlay.

The point of using it in this project: it is our measurement of whether the
network looks at the leaf lesion or at the background.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config
from torch_data import denormalize


class GradCAM:
    """
    Usage:
        cam_extractor = GradCAM(model, target_layer)
        cam = cam_extractor(images, class_idx=None)   # (B, H, W) in [0, 1]
        cam_extractor.remove()                        # detach the hooks
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Forward hook captures the feature maps A^k on the way in;
        # the full backward hook captures dy^c/dA^k on the way out.
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        images: torch.Tensor,
        class_idx: torch.Tensor | int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (cams, class_indices):
            cams          -- (B, H, W) float array in [0, 1], H/W = input size
            class_indices -- (B,) the class each heatmap explains
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Grad-CAM needs gradients, so this must not run under no_grad, and it
        # is kept in fp32 even when training used mixed precision.
        images = images.clone().requires_grad_(False)
        logits = self.model(images)

        if class_idx is None:
            target = logits.argmax(dim=1)
        elif isinstance(class_idx, int):
            target = torch.full((images.size(0),), class_idx,
                                dtype=torch.long, device=logits.device)
        else:
            target = class_idx.to(logits.device)

        # Sum of the target-class scores: because each sample contributes to
        # only its own score, the per-sample gradients stay independent.
        score = logits.gather(1, target.unsqueeze(1)).sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "No activations captured. The target layer is probably not on "
                "the forward path of this model."
            )

        # alpha_k = global average pool of the gradients over the spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # (B, K, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=images.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        # Normalise each heatmap to [0, 1] independently; the absolute scale of
        # a CAM is not meaningful, only the relative spatial pattern is.
        flat = cam.flatten(1)
        cam_min = flat.min(dim=1).values.view(-1, 1, 1)
        cam_max = flat.max(dim=1).values.view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu().numpy(), target.detach().cpu().numpy()

    def remove(self) -> None:
        """Always call this: hooks left attached leak memory across runs."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def get_target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    """
    The last convolutional stage of each backbone -- the deepest layer that
    still has spatial structure, which is exactly what Grad-CAM needs.
    """
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer configured for '{arch}'")


def save_cam_grid(
    images: torch.Tensor,
    cams: np.ndarray,
    true_labels: list[str],
    pred_labels: list[str],
    out_name: str,
    alpha: float = 0.45,
) -> Path:
    """
    Two rows per example: the photograph, and the photograph with its heatmap.
    This figure is the qualitative evidence for the 'where does it look'
    question in the report.
    """
    config.ensure_dirs()
    n = min(len(cams), 8)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        photo = denormalize(images[i])
        axes[0, i].imshow(photo)
        axes[0, i].set_axis_off()
        correct = true_labels[i] == pred_labels[i]
        axes[0, i].set_title(
            f"true: {true_labels[i][:18]}\npred: {pred_labels[i][:18]}",
            fontsize=7, color="green" if correct else "red",
        )

        axes[1, i].imshow(photo)
        axes[1, i].imshow(cams[i], cmap="jet", alpha=alpha)
        axes[1, i].set_axis_off()

    fig.suptitle("Grad-CAM: top row original, bottom row class evidence", fontsize=10)
    fig.tight_layout()
    path = config.FIG_DIR / f"{out_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[gradcam] saved figure -> {path}")
    return path
