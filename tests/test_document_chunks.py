"""Tests for document chunk ORM mapping."""

from meipi.indexing.embedding.text_preprocess import ChunkConfig, DocumentChunker
from meipi.indexing.model import DBBgeM3Vector


def test_chunk_doc_maps_to_bge_m3_rows() -> None:
    chunker = DocumentChunker(ChunkConfig(clean_text=False))
    chunks = chunker.chunk_doc(7, "Vertragstext")
    rows = [
        DBBgeM3Vector(
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
        )
        for chunk in chunks
    ]
    assert len(rows) == 1
    assert rows[0].doc_id == 7
    assert rows[0].chunk_index == 0
    assert rows[0].content == "Vertragstext"
    assert rows[0].vector is None
    assert isinstance(rows[0], DBBgeM3Vector)


def test_chunk_doc_preserves_order_in_rows() -> None:
    chunker = DocumentChunker(
        ChunkConfig(clean_text=False, max_tokens=32, overlap=0, min_chunk_tokens=1)
    )
    text = ("Alpha paragraph. " * 40) + "\n\n" + ("Beta paragraph. " * 40)
    chunks = chunker.chunk_doc(5, text)
    rows = [
        DBBgeM3Vector(
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
        )
        for chunk in chunks
    ]
    assert len(rows) >= 2
    assert [row.chunk_index for row in rows] == list(range(len(rows)))
    assert all(isinstance(row, DBBgeM3Vector) for row in rows)


def test_bge_m3_vector_size() -> None:
    assert DBBgeM3Vector._vector_size() == 1024
