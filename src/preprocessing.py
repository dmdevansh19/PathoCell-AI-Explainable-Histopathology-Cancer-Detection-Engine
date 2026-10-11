"""Stain normalization and patch extraction for histopathology images."""
import numpy as np
from PIL import Image


def normalize_stain(image: Image.Image) -> Image.Image:
    """Channel-wise histogram stretching (lightweight stain normalization)."""
    arr = np.array(image.convert("RGB")).astype(np.float32)
    for c in range(3):
        channel = arr[:, :, c]
        p2, p98 = np.percentile(channel, (2, 98))
        if p98 > p2:
            arr[:, :, c] = np.clip((channel - p2) / (p98 - p2) * 255.0, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def extract_patches(image: Image.Image, patch_size: int = 224, stride: int = None):
    stride = stride or patch_size
    w, h = image.size
    patches = []
    for top in range(0, max(h - patch_size, 0) + 1, stride):
        for left in range(0, max(w - patch_size, 0) + 1, stride):
            patch = image.crop((left, top, left + patch_size, top + patch_size))
            if patch.size == (patch_size, patch_size):
                patches.append(patch)
    if not patches:
        patches = [image.resize((patch_size, patch_size))]
    return patches


def preprocess_for_model(image: Image.Image, patch_size: int = 224) -> Image.Image:
    return normalize_stain(image).resize((patch_size, patch_size))
