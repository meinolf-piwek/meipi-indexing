"""Tests for DBOperations session usage and updates."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meipi.indexing.model import DBMeta, DBPic
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

    assert ops.docroot == ""
    assert ops.pool_id is None


def test_update_thumbs_updates_rows(mock_db_operations):
    ops, session = mock_db_operations
    thumb = np.zeros((4, 4, 3), dtype=np.uint8)
    meta = MagicMock()
    session.get.return_value = meta

    ops.update_thumbs([(thumb, 10)])

    session.get.assert_called_once_with(DBMeta, 10)
    assert meta.thumbarray is thumb
    meta.set_phash.assert_called_once()
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


@patch("meipi.indexing.document_thumbnail.make_document_thumbnail")
def test_update_thumbs_doc_generates_missing_doc_thumbs(
    mock_make_document_thumbnail, mock_db_operations
):
    ops, session = mock_db_operations
    thumb = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_make_document_thumbnail.return_value = thumb
    session.execute.return_value = [
        SimpleNamespace(id=11, path="docs/a.txt"),
        SimpleNamespace(id=12, path="docs/b.pdf"),
    ]

    with patch.object(ops, "update_thumbs") as mock_update_thumbs:
        failed = ops.update_thumbs_doc()

    assert failed == []
    assert mock_make_document_thumbnail.call_count == 2
    assert mock_update_thumbs.call_count == 2
    mock_update_thumbs.assert_any_call([(thumb, 11)])
    mock_update_thumbs.assert_any_call([(thumb, 12)])


def test_update_phashes_computes_missing_hashes(mock_db_operations):
    ops, session = mock_db_operations
    meta = MagicMock()
    session.scalars.return_value.all.return_value = [meta]

    count = ops.update_phashes()

    assert count == 1
    meta.set_phash.assert_called_once()
    session.flush.assert_called_once()
    session.commit.assert_called_once()
