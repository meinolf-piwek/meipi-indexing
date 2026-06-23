"""Search implementations for indexed documents."""

from __future__ import annotations

from .classic_search import (
    DocSearchHit,
    DocSearchResult,
    QueryMode,
    SortField,
    search_documents,
)

__all__ = [
    "DocSearchHit",
    "DocSearchResult",
    "QueryMode",
    "SortField",
    "search_documents",
]
