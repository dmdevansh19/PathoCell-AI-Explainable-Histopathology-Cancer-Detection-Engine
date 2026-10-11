"""Loads the model once and exposes predict() for the Streamlit app."""
import sys
import os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.model import load_trained_model, CLASS_NAMES
from src.preprocessing import normalize_stain
from src.explainability import generate_gradcam
from src.utils import get_device, format_confidence

DEVICE = get_device()
WEIGHTS_PATH = os.path.join(ROOT, "models", "best_model.pt")
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_trained_model(WEIGHTS_PATH, backbone="resnet50", device=DEVICE)
    return _model


def predict(image: Image.Image):
    model = _get_model()
    pred_idx, confidence, heatmap = generate_gradcam(model, normalize_stain(image), device=DEVICE)
    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": format_confidence(confidence),
        "confidence_raw": confidence,
        "heatmap": heatmap,
    }
