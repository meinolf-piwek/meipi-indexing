"""Tests for PIL-only thumbnail generation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from meipi.indexing.thumbnail import THUMB_SIZE, make_thumbnail_array


def test_make_thumbnail_array_produces_square_rgb(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (400, 200), color=(255, 0, 0)).save(path)

    thumb = make_thumbnail_array(str(path))

    assert thumb.shape == (THUMB_SIZE, THUMB_SIZE, 3)
    assert thumb.dtype == np.uint8


def test_make_thumbnail_array_preserves_aspect_ratio_padding(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (400, 200), color=(0, 255, 0)).save(path)

    thumb = make_thumbnail_array(str(path))

    assert thumb[:, :, 1].max() == 255
    assert thumb[0, 0, 1] == 0
    assert thumb[-1, 0, 1] == 0
