"""Tests for document text chunking."""

from meipi.indexing.embedding.text_preprocess import ChunkConfig, DocumentChunker
from meipi.indexing.model import ChunkItem


def test_chunk_doc_returns_chunk_items() -> None:
    chunker = DocumentChunker(ChunkConfig(clean_text=False))
    chunks = chunker.chunk_doc(7, "Alpha paragraph.\n\nBeta paragraph.")
    assert len(chunks) >= 1
    assert isinstance(chunks[0], ChunkItem)
    assert chunks[0].doc_id == 7
    assert chunks[0].chunk_index == 0
    assert chunks[0].content
