"""Tests for DBOperations session usage and updates."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meipi.indexing.model import DBPic
from meipi.indexing.operations import DBOperations


def test_db_operations_requires_pool_by_default(monkeypatch):
    monkeypatch.setattr("meipi.indexing.operations.sa.create_engine", lambda *a, **k: MagicMock())
    monkeypatch.setattr("meipi.indexing.operations.sessionmaker", lambda **kw: MagicMock())

    with pytest.raises(ValueError, match="Either pool_id or pool must be provided"):
        DBOperations()


def test_db_operations_allow_no_pool(monkeypatch):
    monkeypatch.setattr("meipi.indexing.operations.sa.create_engine", lambda *a, **k: MagicMock())
    monkeypatch.setattr("meipi.indexing.operations.sessionmaker", lambda **kw: MagicMock())

    ops = DBOperations(allow_no_pool=True)

    assert ops.pool.rootpath == ""
    assert ops.docroot == ""


def test_update_thumbs_updates_rows(mock_db_operations):
    ops, session = mock_db_operations
    thumb = np.zeros((4, 4, 3), dtype=np.uint8)
    pic = MagicMock()
    session.get.return_value = pic

    ops.update_thumbs([(thumb, 10)])

    session.get.assert_called_once_with(DBPic, 10)
    assert pic.thumbarray is thumb
    pic.set_phash.assert_called_once()
    session.flush.assert_called_once()
    session.commit.assert_called_once()


@patch("meipi.indexing.operations._dali_resizer")
def test_update_thumbs_no_heic_queries_and_delegates(mock_resizer_cls, mock_db_operations):
    ops, session = mock_db_operations
    session.execute.return_value = [SimpleNamespace(id=5, path="scan.png")]
    mock_resizer_cls.return_value.resize_pics.return_value = ([], [], [], [])

    result = ops.update_thumbs_no_heic()

    session.execute.assert_called_once()
    assert session.__enter__.called
    mock_resizer_cls.return_value.resize_pics.assert_called_once_with(
        [(os.path.join(ops.docroot, "scan.png"), 5)],
        batch_size=100,
        use_PIL=False,
    )
    assert result == []


@patch("meipi.indexing.operations._dali_resizer")
def test_update_thumbs_no_thumb_queries_and_delegates(mock_resizer_cls, mock_db_operations):
    ops, session = mock_db_operations
    session.execute.return_value = [
        SimpleNamespace(id=1, path="a.jpg"),
        SimpleNamespace(id=2, path="b.jpg"),
    ]
    mock_resizer_cls.return_value.resize_pics.return_value = ([], [], [], [])

    result = ops.update_thumbs_no_thumb()

    session.execute.assert_called_once()
    assert session.__enter__.called
    mock_resizer_cls.return_value.resize_pics.assert_called_once_with(
        [
            (os.path.join(ops.docroot, "a.jpg"), 1),
            (os.path.join(ops.docroot, "b.jpg"), 2),
        ],
        batch_size=1,
        use_PIL=True,
    )
    assert result == []
