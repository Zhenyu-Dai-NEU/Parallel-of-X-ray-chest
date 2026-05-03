"""
DenseNet-121 model for multi-label thoracic disease classification.
Uses pretrained ImageNet weights with modified final classifier.
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121ChestXray(nn.Module):
    """DenseNet-121 adapted for 14-class multi-label chest X-ray classification."""

    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        # Load pretrained DenseNet-121
        if pretrained:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.model = models.densenet121(weights=None)

        # Replace classifier for multi-label output
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def get_model(num_classes=14, pretrained=True):
    """Factory function to create the model."""
    return DenseNet121ChestXray(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
