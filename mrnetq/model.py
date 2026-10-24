"""MRNet-style plane model: CNN features, average pooling per slice, max over slices, linear layer."""
import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights, MobileNet_V3_Small_Weights, alexnet, mobilenet_v3_small


class MRNetPlane(nn.Module):
    def __init__(self, backbone='alexnet', pretrained=True):
        super().__init__()
        if backbone == 'alexnet':
            self.features = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1 if pretrained else None).features
            dim = 256
        elif backbone == 'mobilenet_v3_small':
            self.features = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None).features
            dim = 576
        else:
            raise ValueError(backbone)
        self.backbone, self.dim = backbone, dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dim, 1)

    def slice_features(self, x):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return self.gap(self.features(x)).flatten(1)

    def forward(self, x):
        return self.classifier(self.slice_features(x).max(0).values)


class GrayFeatures(nn.Module):
    """CNN part for export; the first convolution takes one gray channel (RGB weights summed)."""

    def __init__(self, m: MRNetPlane):
        super().__init__()
        feats = list(m.features.children())
        first = feats[0] if m.backbone == 'alexnet' else feats[0][0]
        conv = nn.Conv2d(1, first.out_channels, first.kernel_size, first.stride, first.padding,
                         bias=first.bias is not None)
        with torch.no_grad():
            conv.weight.copy_(first.weight.sum(1, keepdim=True))
            if first.bias is not None:
                conv.bias.copy_(first.bias)
        if m.backbone == 'alexnet':
            self.features = nn.Sequential(conv, *feats[1:])
        else:
            block0 = nn.Sequential(conv, *list(feats[0].children())[1:])
            self.features = nn.Sequential(block0, *feats[1:])
        self.gap = m.gap

    def forward(self, x):
        return self.gap(self.features(x)).flatten(1)
