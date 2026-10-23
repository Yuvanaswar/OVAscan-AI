"""
Model Architectures for PCOS Ultrasound AI:
- V5 Baseline: EfficientNet-B0 (4.01M parameters)
- V6 Final Champion: ConvNeXt-Tiny (27.82M parameters, 7x7 depthwise convolutions)
"""

import torch
import torch.nn as nn
from torchvision import models

def build_model(arch: str = "convnext_tiny", num_classes: int = 3, pretrained: bool = False):
    """
    Builds the model architecture for 3-class Ovarian Ultrasound Classification:
    - convnext_tiny / v6 (Primary champion: 7x7 depthwise convs)
    - resnet50 (Deep residual baseline)
    - mobilenet_v2 (Ultra-fast mobile baseline)
    - efficientnet_b0 / v5 (Secondary compound scaled baseline)
    """
    arch = arch.lower()
    if arch in ["convnext_tiny", "v6"]:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    elif arch in ["resnet50", "resnet"]:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif arch in ["mobilenet_v2", "mobilenet"]:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif arch in ["efficientnet_b0", "v5"]:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        return model

    else:
        raise ValueError(f"Unsupported architecture: {arch}. Supported: ['convnext_tiny', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']")

