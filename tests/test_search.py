"""Tests for document full-text search."""

from meipi.indexing.search import search_documents
from helpers import POSTGRES_DIALECT


def test_search_documents_empty_query(mock_db_operations):
    ops, session = mock_db_operations

    hits = search_documents(session, pool_id=ops.pool.id, query="   ")

    assert hits == []
    session.execute.assert_not_called()


def test_search_sql_includes_metadata_tsvector(mock_db_operations):
    """Search matches content or metadata tsvector."""
    from sqlalchemy import or_, select

    from meipi.indexing.model import DBMeta
    from meipi.indexing.search import _metadata_tsvector, _tsquery

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
