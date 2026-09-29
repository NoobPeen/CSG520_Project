"""
Approach 3 model: a small VGG-style CNN trained from random initialisation.

Why this shape:
  * No residual connections and no pretrained weights. A3 is the "what does a
    network learn when nobody gave it ImageNet features" arm, so it must be
    plain and must start from scratch.
  * Each block is two 3x3 conv -> BatchNorm -> ReLU layers followed by 2x2
    max-pooling: two stacked 3x3 convs see a 5x5 region with fewer weights
    than one 5x5 conv, and BatchNorm is what makes a 10-layer plain network
    trainable from scratch without careful initialisation tricks.
  * Global average pooling instead of flattening the 7x7x256 map into a big
    fully connected layer. That keeps the head to ~10k weights (the whole
    model is ~2.4M, about a tenth of ResNet50) and keeps the last conv map
    spatially meaningful, which is what Grad-CAM reads.

    Input 224x224x3
    block1  32 ch -> 112x112
    block2  64 ch ->  56x56
    block3 128 ch ->  28x28
    block4 256 ch ->  14x14
    block5 256 ch ->   7x7     <- Grad-CAM target (model.features.block5)
    GAP -> 256 -> Dropout(0.3) -> Linear(256, num_classes)
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

DEFAULT_WIDTHS = (32, 64, 128, 256, 256)


def conv_bn_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    # bias=False: BatchNorm's own shift makes a conv bias redundant.
    # inplace=False keeps Grad-CAM's backward hooks safe.
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),
    ]


def vgg_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        *conv_bn_relu(in_ch, out_ch),
        *conv_bn_relu(out_ch, out_ch),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, widths=DEFAULT_WIDTHS, dropout: float = 0.3) -> None:
        super().__init__()
        blocks, in_ch = OrderedDict(), 3
        for i, width in enumerate(widths, start=1):
            blocks[f"block{i}"] = vgg_block(in_ch, width)
            in_ch = width
        self.features = nn.Sequential(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_ch, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        # He (Kaiming) init suits ReLU networks: it keeps activation variance
        # roughly constant from layer to layer at the start of training.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last conv block (7x7x256 output): deepest layer with spatial layout."""
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = CustomCNN(num_classes=38)
    out = m(torch.randn(2, 3, 224, 224))
    print(m)
    print(f"output {tuple(out.shape)}  params {count_parameters(m):,}")
