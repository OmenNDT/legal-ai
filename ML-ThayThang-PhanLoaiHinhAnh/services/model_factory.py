import torch.nn as nn
import torch
from typing import cast
from torchvision.models import (
    efficientnet_b4, EfficientNet_B4_Weights,
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights
)

from config.GetPath import paths
from utils.config import Config

class ModelFactory:

    SUPPORTED = ("efficientnetb4", "resnet50", "vit")

    @staticmethod
    def build(model_name: str, num_classes: int = Config.NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
        name = model_name.lower()
        print(f"Building {name.upper()} | Freeze_backbone = {freeze_backbone}")

        if name == "efficientnetb4":
            model = efficientnet_b4(weights = EfficientNet_B4_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for p in model.features.parameters():
                    p.requires_grad = False
            in_f = cast(nn.Linear, model.classifier[1]).in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_f, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        elif name == "resnet50":
            model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for p in model.layer1.parameters(): p.requires_grad = False
                for p in model.layer2.parameters(): p.requires_grad = False
            in_f = cast(nn.Linear, model.fc).in_features
            cast(nn.Module, model).fc = nn.Sequential(
                nn.Linear(in_f, 512), 
                nn.BatchNorm1d(512), 
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )

        elif name == "vit":
            model = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_V1)
            if freeze_backbone:
                for p in model.encoder.parameters():
                    p.requires_grad = False
            in_f = cast(nn.Linear, model.heads.head).in_features
            model.heads.head = nn.Sequential(
                nn.Linear(in_f, 256), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        else:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {ModelFactory.SUPPORTED}")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
        return model.to(Config.DEVICE)

    @staticmethod
    def load(model_name: str, num_classes: int = Config.NUM_CLASSES, suffix: str = "") -> nn.Module:
        model_path = paths.models / f"{model_name}{suffix}_best.pt"
        model = ModelFactory.build(model_name, num_classes, freeze_backbone = False)
        model.load_state_dict(torch.load(model_path, map_location = Config.DEVICE, weights_only = True))
        print(f"Loaded <- {model_path}")
        return model
