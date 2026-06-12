"""Tests for document text chunking."""

from meipi.indexing.embedding.text_embedding import ChunkConfig, DocumentChunker


def test_chunk_text_uses_pre_cleaned_inhalt() -> None:
    chunker = DocumentChunker(ChunkConfig(clean_text=False))
    chunks = chunker.chunk_text("Alpha paragraph.\n\nBeta paragraph.", text_id=3)
    assert len(chunks) >= 1
    assert chunks[0]["id"][0] == 3
