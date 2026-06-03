"""Tests for update_thumb_for_pic and update_thumb_for_path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


@patch("meipi.indexing.operations._dali_resizer")
def test_update_thumb_for_pic_calls_resizer_and_update_thumbs(
    mock_dali_resizer, mock_db_operations
):
    dbop, _session = mock_db_operations
    resizer = MagicMock()
    mock_dali_resizer.return_value = resizer
    resizer.resize_pics.return_value = ([np.zeros((224, 224, 3))], [42], [], [])

    with patch.object(dbop, "update_thumbs") as mock_update_thumbs:
        result = dbop.update_thumb_for_pic(42, "pics/a.jpg")

    assert result is True
    resizer.resize_pics.assert_called_once()
    mock_update_thumbs.assert_called_once()


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
