"""Tests for ORM types and small model helpers."""

import numpy as np
from PIL import Image

from meipi.indexing.model import DBMeta, PILArray


def test_pilarray_roundtrip():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[0, 0, 0] = 255
    column = PILArray()

    stored = column.process_bind_param(arr, dialect=None)
    assert isinstance(stored, bytes)

    restored = column.process_result_value(stored, dialect=None)
    np.testing.assert_array_equal(restored, arr)


def test_pilarray_none_roundtrip():
    column = PILArray()
    assert column.process_bind_param(None, dialect=None) is None
    assert column.process_result_value(None, dialect=None) is None


def test_calc_phash_stable_for_identical_images():
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))
    h1 = DBMeta.calc_phash(image)
    h2 = DBMeta.calc_phash(image)
    assert h1 == h2
    assert isinstance(h1, bytes)
    assert len(h1) > 0


def test_dino_vector_size():
    from meipi.indexing.model import DBDinoV2Vector

    assert DBDinoV2Vector._vector_size() == 1024
