"""
models.py
---------
Three CNN architectures for disaster damage assessment.
"""

import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64, 128),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=13, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,    64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 256, pool=False),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet50FineTuned(nn.Module):
    def __init__(self, num_classes=13, dropout=0.5):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self):
        for layer in list(self.backbone.children())[6:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def build_model(model_name, num_classes=13):
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    elif model_name == "improved":
        return ImprovedCNN(num_classes=num_classes)
    elif model_name == "resnet50":
        return ResNet50FineTuned(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")