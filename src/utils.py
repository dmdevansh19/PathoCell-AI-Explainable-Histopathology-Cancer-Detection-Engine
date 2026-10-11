import os
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def format_confidence(confidence: float) -> str:
    return f"{confidence * 100:.1f}%"
