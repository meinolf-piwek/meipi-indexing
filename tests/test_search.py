"""Tests for document full-text search."""

from datetime import datetime
from types import SimpleNamespace

from meipi.indexing.search import DocSearchResult, search_documents
from meipi.indexing.search.classic_search import _metadata_tsvector, _normalize_suffixes, _tsquery
from helpers import POSTGRES_DIALECT


def test_search_documents_empty_query_applies_filters_only(mock_db_operations):
    ops, session = mock_db_operations
    now = datetime(2024, 3, 1, 8, 0, 0)
    session.scalar.return_value = 1
    session.execute.return_value = [
        SimpleNamespace(
            meta_id=7,
            path="docs/a.txt",
            fname="a.txt",
            suffix=".txt",
            sort_date=now,
            snippet="",
            thumbarray=None,
        )
    ]

    result = search_documents(
        session,
        pool_id=ops.pool.id,
        query="   ",
        suffixes=[".txt"],
        path_prefix="docs/",
    )

    assert result.total_count == 1
    assert len(result.hits) == 1
    assert result.hits[0].fname == "a.txt"
    session.scalar.assert_called_once()
    session.execute.assert_called_once()


def test_search_sql_includes_metadata_tsvector(mock_db_operations):
    """Search matches content or metadata tsvector."""
    from sqlalchemy import or_, select

    from meipi.indexing.model import DBMeta

    ops, _session = mock_db_operations
    lang = "german"
    tsq = _tsquery(lang, "vertrag", "plain")
    meta_ts = _metadata_tsvector(lang)

    stmt = (
        select(DBMeta.id)
        .where(DBMeta.pool_id == ops.pool.id)
        .where(
            or_(
                DBMeta.ts_content.bool_op("@@")(tsq),
                meta_ts.bool_op("@@")(tsq),
            )
        )
    )
    sql = str(stmt.compile(dialect=POSTGRES_DIALECT)).lower()
    assert "@@" in sql
    assert "to_tsvector" in sql
    assert "meta_data" in sql


def test_normalize_suffixes_accepts_dotted_and_plain_values():
    assert _normalize_suffixes([".PDF", "txt", " .md "]) == [".pdf", ".txt", ".md"]
    assert _normalize_suffixes([]) == []
    assert _normalize_suffixes(None) == []
