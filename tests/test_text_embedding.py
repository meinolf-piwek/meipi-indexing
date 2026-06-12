"""Tests for document text chunking."""

from meipi.indexing.embedding.text_embedding import ChunkConfig, DocumentChunker


def test_clean_text_preserves_paragraph_breaks() -> None:
    chunker = DocumentChunker(ChunkConfig())
    text = "First   paragraph.\n\nSecond\tparagraph.\nwith wrap"
    cleaned = chunker._clean_text(text)
    assert cleaned == "First paragraph.\n\nSecond paragraph. with wrap"


def test_split_paragraphs_after_cleaning() -> None:
    chunker = DocumentChunker(ChunkConfig())
    text = "Alpha line.\n\nBeta   line.\n\n\nGamma line."
    paragraphs = chunker._split_paragraphs(chunker._clean_text(text))
    assert paragraphs == ["Alpha line.", "Beta line.", "Gamma line."]
