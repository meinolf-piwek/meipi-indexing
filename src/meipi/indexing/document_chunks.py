"""Document chunk records for semantic search indexing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .model import DBBgeM3Vector


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """One text chunk ready for storage or embedding."""

    chunk_index: int
    content: str
    meta_id: int

    def to_row(self, doc_id: int) -> DBBgeM3Vector:
        return DBBgeM3Vector(
            doc_id=doc_id,
            chunk_index=self.chunk_index,
            content=self.content,
        )


def chunks_to_rows(doc_id: int, chunks: Sequence[DocumentChunk]) -> list[DBBgeM3Vector]:
    """Build ORM rows for ``DBBgeM3Vector`` from chunker output."""
    return [chunk.to_row(doc_id) for chunk in chunks]
