"""Integration tests against a real PostgreSQL database."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest
import sqlalchemy as sa
from PIL import Image

from meipi.indexing.model import DBDoc, DBMeta, DBPic
from conftest import seed_pic_meta

pytestmark = pytest.mark.integration


def test_recreate_tables_creates_core_tables(db_ops):
    with db_ops.engine.connect() as conn:
        tables = (
            conn.execute(
                sa.text(
                    """
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public'
                      AND tablename IN (
                        'datapools', 'filemeta', 'documents', 'pictures', 'dino_v2_vectors'
                      )
                    ORDER BY tablename
                    """
                )
            )
            .scalars()
            .all()
        )
    assert tables == [
        "datapools",
        "dino_v2_vectors",
        "documents",
        "filemeta",
        "pictures",
    ]


def _paths_without_thumb(db_ops) -> set[str]:
    stmt = (
        sa.select(DBPic.id, DBMeta.path)
        .join(DBPic.meta)
        .where(DBMeta.pool_id == db_ops.pool.id)
        .where(DBPic.thumbarray.is_(None))
    )
    with db_ops.Session() as session:
        return {
            os.path.join(db_ops.docroot, row.path)
            for row in session.execute(stmt)
        }


def _paths_without_thumb_no_heic(db_ops) -> set[str]:
    stmt = (
        sa.select(DBPic.id, DBMeta.path)
        .join(DBPic.meta)
        .where(DBMeta.pool_id == db_ops.pool.id)
        .where(DBPic.thumbarray.is_(None))
        .where(DBMeta.suffix.not_in([".HEIC", ".heic"]))
    )
    with db_ops.Session() as session:
        return {
            os.path.join(db_ops.docroot, row.path)
            for row in session.execute(stmt)
        }


def test_thumb_queries_filter_by_pool_and_heic(db_ops):
    seed_pic_meta(db_ops, path="a.jpg", suffix=".jpg")
    seed_pic_meta(db_ops, path="b.heic", suffix=".heic")
    seed_pic_meta(db_ops, path="c.png", suffix=".png", with_thumb=True)

    no_thumb = _paths_without_thumb(db_ops)
    assert no_thumb == {
        os.path.join(db_ops.docroot, "a.jpg"),
        os.path.join(db_ops.docroot, "b.heic"),
    }

    no_heic = _paths_without_thumb_no_heic(db_ops)
    assert no_heic == {os.path.join(db_ops.docroot, "a.jpg")}


def test_update_thumbs_persists_numpy_thumbnail_and_phash(db_ops):
    _, pic = seed_pic_meta(db_ops, path="thumb.jpg", suffix=".jpg")
    thumb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)

    db_ops.update_thumbs([(thumb, pic.id)])

    with db_ops.Session() as session:
        stored = session.get(DBPic, pic.id)
        assert stored is not None
        assert stored.thumbarray is not None
        np.testing.assert_array_equal(stored.thumbarray, thumb)
        assert stored.phash is not None


def test_insert_pics_from_meta_selects_only_missing_pics(db_ops, tmp_path, monkeypatch):
    """Meta rows without pictures are returned; existing picture rows are excluded."""
    seed_pic_meta(db_ops, path="has_pic.jpg", suffix=".jpg")
    now = datetime(2024, 2, 1, 9, 0, 0)
    with db_ops.Session() as session:
        pending_meta = DBMeta(
            pool_id=db_ops.pool.id,
            path="pending.jpg",
            fname="pending.jpg",
            suffix=".jpg",
            sort_date=now,
            fdate=now,
            fsize=10,
            clength=10,
            ctype="image/jpeg",
            ftype="pic",
            md_keys=[],
            meta_data={},
            sha256=b"\xcd" * 32,
        )
        session.add(pending_meta)
        session.commit()
        session.refresh(pending_meta)

    pending_file = tmp_path / "files" / "pending.jpg"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(pending_file)

    monkeypatch.setattr(
        "meipi.indexing.operations.xmpu.file_to_dict",
        lambda path: {},
    )

    with db_ops.Session() as session:
        subq = sa.select(DBPic.id).where(DBPic.meta_id == DBMeta.id).exists()
        stmt = sa.select(DBMeta).where(
            DBMeta.pool_id == db_ops.pool.id,
            DBMeta.ftype == "pic",
            ~subq,
        )
        pending = session.execute(stmt).scalars().all()

    assert len(pending) == 1
    assert pending[0].path == "pending.jpg"

    db_ops.insert_pics_from_meta()

    with db_ops.Session() as session:
        pic = session.scalar(
            sa.select(DBPic).where(DBPic.meta_id == pending_meta.id)
        )
        assert pic is not None
        assert pic.xmp == {}
        assert pic.meta_id == pending_meta.id


def test_dbdoc_fulltext_roundtrip(db_ops):
    now = datetime(2024, 3, 1, 8, 0, 0)
    with db_ops.Session() as session:
        meta = DBMeta(
            pool_id=db_ops.pool.id,
            path="notes.txt",
            fname="notes.txt",
            suffix=".txt",
            sort_date=now,
            fdate=now,
            fsize=20,
            clength=20,
            ctype="text/plain",
            ftype="doc",
            md_keys=[],
            meta_data={},
            sha256=b"\xef" * 32,
        )
        meta.inhalt = "Der schnelle braune Fuchs springt über den lazy Hund"
        meta.doc = DBDoc()
        session.add(meta)
        session.commit()
        session.refresh(meta)
        meta_id = meta.id

    with db_ops.Session() as session:
        hits = DBMeta.tsquery("Fuchs", session, lang="german")
    assert len(hits) == 1
    assert hits[0].id == meta_id


def test_search_documents_ranked_hits(db_ops):
    from meipi.indexing.search import search_documents

    now = datetime(2024, 3, 1, 8, 0, 0)
    with db_ops.Session() as session:
        meta = DBMeta(
            pool_id=db_ops.pool.id,
            path="report.txt",
            fname="report.txt",
            suffix=".txt",
            sort_date=now,
            fdate=now,
            fsize=40,
            clength=40,
            ctype="text/plain",
            ftype="doc",
            md_keys=[],
            meta_data={},
            sha256=b"\xab" * 32,
        )
        meta.inhalt = "Der Jahresbericht enthält wichtige Vertragsdetails."
        meta.doc = DBDoc()
        session.add(meta)
        session.commit()

    with db_ops.Session() as session:
        hits = search_documents(
            session,
            pool_id=db_ops.pool.id,
            query="Vertrag",
            lang="german",
            mode="plain",
        )

    assert len(hits) == 1
    assert hits[0].fname == "report.txt"
    assert hits[0].rank > 0
    assert "Vertrag" in hits[0].snippet or "vertrag" in hits[0].snippet.lower()


def test_search_documents_matches_metadata_only(db_ops):
    from meipi.indexing.search import search_documents

    now = datetime(2024, 3, 1, 8, 0, 0)
    with db_ops.Session() as session:
        meta = DBMeta(
            pool_id=db_ops.pool.id,
            path="scan.tiff",
            fname="scan.tiff",
            suffix=".tiff",
            sort_date=now,
            fdate=now,
            fsize=10,
            clength=10,
            ctype="image/tiff",
            ftype="pic",
            md_keys=["title"],
            meta_data={"title": "Vertragsentwurf Sommer 2024"},
            sha256=b"\xcd" * 32,
            inhalt="",
        )
        session.add(meta)
        session.commit()

    with db_ops.Session() as session:
        hits = search_documents(
            session,
            pool_id=db_ops.pool.id,
            query="Vertragsentwurf",
            lang="german",
            mode="plain",
        )

    assert len(hits) == 1
    assert hits[0].fname == "scan.tiff"
