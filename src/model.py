"""CNN classifier (transfer learning) for benign vs malignant classification."""
import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = ["benign", "malignant"]


def build_model(backbone: str = "resnet50", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def load_trained_model(weights_path: str, backbone: str = "resnet50", device: str = "cpu") -> nn.Module:
    model = build_model(backbone=backbone, pretrained=True)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded fine-tuned weights from {weights_path}")
    except FileNotFoundError:
        print(f"No trained weights at {weights_path} -- using ImageNet backbone only (placeholder predictions).")
    model.to(device)
    model.eval()
    return model
