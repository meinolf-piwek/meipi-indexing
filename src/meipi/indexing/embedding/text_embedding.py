import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from transformers import AutoTokenizer


@dataclass
class ChunkConfig:
    model_name: str = "BAAI/bge-m3"
    max_tokens: int = 512
    overlap: int = 80
    min_chunk_tokens: int = 50
    clean_text: bool = True


class DocumentChunker:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # ------------------------
    # Public API
    # ------------------------

    def chunk_text(self, text: str, text_id: int = 0) -> list[dict[str, Any]]:
        if self.config.clean_text:
            text = self._clean_text(text)

        paragraphs = self._split_paragraphs(text)
        chunks = self._build_chunks(paragraphs)

        return [
            {
                "text": chunk,
                "id": (text_id, i)
            }
            for i, chunk in enumerate(chunks)
        ]

    # ------------------------
    # Cleaning
    # ------------------------

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    # ------------------------
    # Splitting
    # ------------------------

    def _split_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]

    # ------------------------
    # Chunk Builder
    # ------------------------

    def _build_chunks(self, paragraphs: list[str]) -> list[str]:
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # Falls einzelner Absatz zu groß → hart splitten
            if para_tokens > self.config.max_tokens:
                chunks.extend(self._split_long_text(para))
                continue

            if current_tokens + para_tokens <= self.config.max_tokens:
                current_chunk.append(para)
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Overlap hinzufügen
        return self._apply_overlap(chunks)

    # ------------------------
    # Token Handling
    # ------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_long_text(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        step = self.config.max_tokens - self.config.overlap

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.config.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            if len(chunk_tokens) >= self.config.min_chunk_tokens:
                chunks.append(chunk_text)

        return chunks

    # ------------------------
    # Overlap
    # ------------------------

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap to the chunks."""
        if self.config.overlap <= 0:
            return chunks

        overlapped = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
                continue

            prev_tokens = self.tokenizer.encode(
                chunks[i - 1],
                add_special_tokens=False
            )

            overlap_tokens = prev_tokens[-self.config.overlap:]
            overlap_text = self.tokenizer.decode(overlap_tokens)

            merged = overlap_text + " " + chunk
            overlapped.append(merged)

        return overlapped

class BulkDocumentChunker:
    def __init__(self, chunker: DocumentChunker):
        self.chunker = chunker

    def batch_chunk_text(
        self, text_list: list[tuple[str, int]]
    ) -> Iterator[dict[str, Any]]:
        for text, text_id in text_list:
            yield from self.chunker.chunk_text(text, text_id)