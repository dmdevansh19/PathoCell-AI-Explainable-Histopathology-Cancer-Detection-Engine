"""Grad-CAM and occlusion-based (SHAP-style) explanations."""
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _get_target_layer(model):
    if hasattr(model, "layer4"):
        return [model.layer4[-1]]
    if hasattr(model, "features"):
        return [model.features[-1]]
    raise ValueError("Could not locate a target conv layer for this backbone.")


def generate_gradcam(model, image: Image.Image, device: str = "cpu"):
    """Returns (pred_idx, confidence, heatmap_overlay PIL image)."""
    img = image.convert("RGB").resize((224, 224))
    rgb = np.array(img).astype(np.float32) / 255.0
    x = _to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    with GradCAM(model=model, target_layers=_get_target_layer(model)) as cam:
        gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
    overlay = show_cam_on_image(rgb, gray, use_rgb=True)
    return pred_idx, confidence, Image.fromarray(overlay)


def generate_shap_summary(model, image: Image.Image, device: str = "cpu", grid: int = 8):
    """Occlusion attribution (lightweight SHAP-style). Returns grid x grid importance map in [0,1]."""
    img = image.convert("RGB").resize((224, 224))
    x = _to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        base = torch.softmax(model(x), dim=1)[0]
        pred_idx = int(torch.argmax(base).item())
        base_conf = float(base[pred_idx].item())

    cell = 224 // grid
    imp = np.zeros((grid, grid), dtype=np.float32)
    arr = np.array(img).astype(np.uint8)
    for i in range(grid):
        for j in range(grid):
            occ = arr.copy()
            occ[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = 128
            t = _to_tensor(Image.fromarray(occ)).unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.softmax(model(t), dim=1)[0]
            imp[i, j] = base_conf - float(p[pred_idx].item())
    return (imp - imp.min()) / (np.ptp(imp) + 1e-8)
