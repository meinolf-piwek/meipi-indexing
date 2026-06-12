"""Tests for document text chunking."""

from meipi.indexing.document_chunks import DocumentChunk
from meipi.indexing.embedding.text_embedding import ChunkConfig, DocumentChunker


def test_chunk_text_returns_document_chunks() -> None:
    chunker = DocumentChunker(ChunkConfig(clean_text=False))
    chunks = chunker.chunk_text("Alpha paragraph.\n\nBeta paragraph.", meta_id=3)
    assert len(chunks) >= 1
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].meta_id == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].content
