"""Tests for document text chunking."""

from meipi.indexing.embedding.text_preprocess import ChunkConfig, DocumentChunker


def test_chunk_text_returns_text_chunks() -> None:
    chunker = DocumentChunker(ChunkConfig(clean_text=False))
    chunks = chunker.chunk_text("Alpha paragraph.\n\nBeta paragraph.")
    assert len(chunks) >= 1
    assert isinstance(chunks[0], str)
    assert chunks[0]
