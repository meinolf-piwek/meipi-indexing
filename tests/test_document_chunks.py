"""Tests for document chunk ORM mapping."""

from meipi.indexing.embedding.text_preprocess import chunks_to_rows
from meipi.indexing.model import DBBgeM3Vector


def test_chunks_to_rows_builds_orm_rows() -> None:
    rows = chunks_to_rows(7, ["Vertragstext"])
    assert len(rows) == 1
    assert rows[0].doc_id == 7
    assert rows[0].chunk_index == 0
    assert rows[0].content == "Vertragstext"
    assert rows[0].vector is None
    assert isinstance(rows[0], DBBgeM3Vector)


def test_chunks_to_rows_preserves_order() -> None:
    rows = chunks_to_rows(5, ["A", "B"])
    assert len(rows) == 2
    assert [row.chunk_index for row in rows] == [0, 1]
    assert all(isinstance(row, DBBgeM3Vector) for row in rows)


def test_bge_m3_vector_size() -> None:
    assert DBBgeM3Vector._vector_size() == 1024
