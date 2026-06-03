"""PIL-only thumbnail generation without CUDA or DALI."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFile
from pillow_heif import register_heif_opener

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True

THUMB_SIZE = 224


def make_thumbnail_array(filepath: str, size: int = THUMB_SIZE) -> np.ndarray:
    """Load an image and return a square RGB thumbnail as a numpy array."""
    with Image.open(filepath) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized = rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
        padded = Image.new("RGB", (size, size))
        padded.paste(
            resized,
            ((size - new_width) // 2, (size - new_height) // 2),
        )
        return np.asarray(padded)
