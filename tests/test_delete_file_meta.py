"""Tests for delete_file_meta on DBOperations."""

from __future__ import annotations

from meipi.indexing.model import DBMeta


def test_delete_file_meta_issues_delete_and_commits(mock_db_operations):
    dbop, session = mock_db_operations
    query = session.query.return_value
    filter_result = query.filter.return_value
    filter_result.delete.return_value = 1

    deleted = dbop.delete_file_meta("docs/a.txt")

    assert deleted is True
    session.query.assert_called_once_with(DBMeta)
    query.filter.assert_called_once()
    filter_result.delete.assert_called_once()
    session.commit.assert_called_once()


def test_delete_file_meta_returns_false_when_missing(mock_db_operations):
    dbop, session = mock_db_operations
    query = session.query.return_value
    filter_result = query.filter.return_value
    filter_result.delete.return_value = 0

    deleted = dbop.delete_file_meta("missing.txt")

    assert deleted is False
    session.commit.assert_called_once()
