"""Tests for document chunk records and ORM mapping."""

from meipi.indexing.document_chunks import DocumentChunk, chunks_to_rows
from meipi.indexing.model import DBBgeM3Vector


def test_document_chunk_to_row() -> None:
    chunk = DocumentChunk(chunk_index=2, content="Vertragstext", meta_id=42)
    row = chunk.to_row(doc_id=7)
    assert row.doc_id == 7
    assert row.chunk_index == 2
    assert row.content == "Vertragstext"
    assert row.vector is None


def test_chunks_to_rows_preserves_order() -> None:
    chunks = [
        DocumentChunk(chunk_index=0, content="A", meta_id=1),
        DocumentChunk(chunk_index=1, content="B", meta_id=1),
    ]
    rows = chunks_to_rows(5, chunks)
    assert len(rows) == 2
    assert [row.chunk_index for row in rows] == [0, 1]
    assert all(isinstance(row, DBBgeM3Vector) for row in rows)


def test_bge_m3_vector_size() -> None:
    assert DBBgeM3Vector._vector_size() == 1024
