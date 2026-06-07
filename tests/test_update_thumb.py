"""Tests for update_thumb_for_pic and update_thumb_for_path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


@patch("meipi.indexing.thumbnail.make_thumbnail_array")
def test_update_thumb_for_pic_calls_pil_and_update_thumbs(
    mock_make_thumbnail, mock_db_operations
):
    dbop, session = mock_db_operations
    thumb = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_make_thumbnail.return_value = thumb
    pic = MagicMock(meta_id=7)
    session.get.return_value = pic

    with patch.object(dbop, "update_thumbs") as mock_update_thumbs:
        result = dbop.update_thumb_for_pic(42, "pics/a.jpg")

    assert result is True
    mock_make_thumbnail.assert_called_once()
    mock_update_thumbs.assert_called_once_with([(thumb, 7)])


def test_update_thumb_for_path_skips_lookup_when_no_row(mock_db_operations):
    dbop, session = mock_db_operations
    session.scalars.return_value.first.return_value = None

    result = dbop.update_thumb_for_path("missing.jpg")

    assert result is False


def test_update_thumb_for_path_delegates_to_update_thumb_for_pic(mock_db_operations):
    dbop, session = mock_db_operations
    session.scalars.return_value.first.return_value = 7

    with patch.object(dbop, "update_thumb_for_pic", return_value=True) as mock_for_pic:
        result = dbop.update_thumb_for_path("pics/a.jpg")

    assert result is True
    mock_for_pic.assert_called_once_with(7, "pics/a.jpg")
