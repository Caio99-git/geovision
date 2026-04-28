"""ResNet50 modified for 13-channel Sentinel-2 input and 10-class EuroSAT output.

The first conv is replaced with a 13-channel version. ImageNet-pretrained RGB
weights are preserved exactly in the first 3 channels; the remaining 10
channels are initialized to the channel-mean of the RGB weights so they start
from a sensible operating point rather than random noise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_13ch(num_classes: int = 10) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=13,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)         # [out, 1, k, k]
        new_w = rgb_mean.repeat(1, 13, 1, 1)
        new_w[:, :3] = old_conv.weight                               # preserve RGB exactly
        new_conv.weight.copy_(new_w)
    model.conv1 = new_conv

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
