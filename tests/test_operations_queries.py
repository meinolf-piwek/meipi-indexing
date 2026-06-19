"""Tests for SQL statement shape used by DBOperations."""

import sqlalchemy as sa

from meipi.indexing.model import DBBgeM3Vector, DBMeta, DBPic

from helpers import compile_postgres


def _meta_without_child_stmt(*, pool_id: int, ftype: str, child_model):
    """Mirror the EXISTS pattern used in insert_pics_from_meta."""
    subq = sa.select(child_model.id).where(child_model.meta_id == DBMeta.id).exists()
    return sa.select(DBMeta).where(
        DBMeta.pool_id == pool_id,
        DBMeta.ftype == ftype,
        ~subq,
    )


def test_insert_missing_chunks_stmt_selects_filemeta_without_vectors():
    has_chunks = (
        sa.select(DBBgeM3Vector.doc_id)
        .where(DBBgeM3Vector.doc_id == DBMeta.id)
        .correlate(DBMeta)
        .exists()
    )
    stmt = sa.select(DBMeta).where(
        DBMeta.pool_id == 7,
        DBMeta.inhalt != "",
        ~has_chunks,
    )
    sql = compile_postgres(stmt).lower()
    assert "filemeta.pool_id" in sql
    assert "7" in sql
    assert "bge_m3_vectors" in sql
    assert "inhalt" in sql
    assert "exists" in sql


def test_insert_pics_stmt_filters_ftype_pool_and_meta_id():
    stmt = _meta_without_child_stmt(pool_id=3, ftype="pic", child_model=DBPic)
    sql = compile_postgres(stmt).lower()
    assert "filemeta.ftype" in sql
    assert "'pic'" in sql
    assert "filemeta.pool_id" in sql
    assert "3" in sql
    assert "exists" in sql
    assert "meta_id" in sql
    assert "pictures.id = filemeta.id" not in sql


def test_python_and_must_not_be_used_for_pic_filters():
    """Regression: plain `and` only keeps the first SQL expression."""
    broken = DBMeta.pool_id == 3 and DBMeta.ftype == "pic"
    sql = compile_postgres(broken).lower()
    assert "ftype" not in sql
    assert "pool_id" in sql


def test_update_thumbs_no_thumb_stmt_filters_by_pool():
    stmt = (
        sa.select(DBMeta.id, DBMeta.path)
        .join(DBMeta.pic)
        .where(DBMeta.pool_id == 5)
        .where(DBMeta.thumbarray.is_(None))
    )
    sql = compile_postgres(stmt).lower()
    assert "filemeta.pool_id" in sql
    assert "5" in sql
    assert "thumbarray" in sql


def test_update_thumbs_no_heic_stmt_excludes_heic():
    stmt = (
        sa.select(DBMeta.id, DBMeta.path)
        .join(DBMeta.pic)
        .where(DBMeta.pool_id == 5)
        .where(DBMeta.thumbarray.is_(None))
        .where(DBMeta.suffix.not_in([".HEIC", ".heic"]))
    )
    sql = compile_postgres(stmt).lower()
    assert "filemeta.pool_id" in sql
    assert "5" in sql
    assert "thumbarray" in sql
    assert ".heic" in sql
