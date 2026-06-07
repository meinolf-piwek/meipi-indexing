"""PostgreSQL full-text search for indexed documents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections.abc import Sequence
from typing import Literal

import numpy as np
import sqlalchemy as sa
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import ColumnElement

from .model import DBMeta

QueryMode = Literal["plain", "websearch", "phrase"]
SortField = Literal["sort_date", "path"]


@dataclass(frozen=True, slots=True)
class DocSearchHit:
    """One filemeta row matching a full-text query."""

    meta_id: int
    path: str
    fname: str
    suffix: str
    sort_date: datetime
    snippet: str
    thumbarray: np.ndarray | None


@dataclass(frozen=True, slots=True)
class DocSearchResult:
    """Full-text search outcome: all matching rows are counted, only ``limit`` are returned."""

    hits: list[DocSearchHit]
    total_count: int


def _tsquery(lang: str, query: str, mode: QueryMode):
    if mode == "plain":
        return func.plainto_tsquery(lang, query)
    if mode == "phrase":
        return func.phaseto_tsquery(lang, query)
    return func.websearch_to_tsquery(lang, query)


def _metadata_text():
    """Plain-text bundle of structural fields and Tika ``meta_data`` JSON."""
    return func.concat(
        DBMeta.fname,
        sa.literal(" "),
        DBMeta.path,
        sa.literal(" "),
        DBMeta.ctype,
        sa.literal(" "),
        func.coalesce(sa.cast(DBMeta.meta_data, sa.Text()), ""),
    )


def _metadata_tsvector(lang: str):
    return func.to_tsvector(lang, _metadata_text())


def _normalize_suffixes(suffixes: Sequence[str] | None) -> list[str]:
    normalized: list[str] = []
    for suffix in suffixes or ():
        value = suffix.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        if value not in normalized:
            normalized.append(value)
    return normalized


def _order_by(sort_by: SortField, sort_desc: bool):
    if sort_by == "path":
        primary = DBMeta.path.desc() if sort_desc else DBMeta.path.asc()
        secondary = DBMeta.sort_date.desc()
    else:
        primary = DBMeta.sort_date.desc() if sort_desc else DBMeta.sort_date.asc()
        secondary = DBMeta.path.asc()
    return primary, secondary


def _text_match(tsq, *, include_metadata: bool, meta_ts):
    content_match = DBMeta.ts_content.bool_op("@@")(tsq)
    if include_metadata:
        meta_match = meta_ts.bool_op("@@")(tsq)
        return or_(content_match, meta_match)
    return content_match


def _search_conditions(
    *,
    pool_id: int,
    tsq=None,
    meta_ts=None,
    include_metadata: bool = True,
    sort_date_from: datetime | None = None,
    sort_date_to: datetime | None = None,
    suffixes: Sequence[str] | None = None,
    path_prefix: str | None = None,
) -> list[ColumnElement[bool]]:
    conditions: list[ColumnElement[bool]] = [DBMeta.pool_id == pool_id]
    if tsq is not None:
        conditions.append(
            _text_match(tsq, include_metadata=include_metadata, meta_ts=meta_ts)
        )
    if sort_date_from is not None:
        conditions.append(DBMeta.sort_date >= sort_date_from)
    if sort_date_to is not None:
        conditions.append(DBMeta.sort_date <= sort_date_to)
    normalized_suffixes = _normalize_suffixes(suffixes)
    if normalized_suffixes:
        conditions.append(DBMeta.suffix.in_(normalized_suffixes))
    prefix = (path_prefix or "").strip()
    if prefix:
        conditions.append(DBMeta.path.startswith(prefix))
    return conditions


def _snippet_expr(lang: str, tsq, *, include_metadata: bool):
    content_snippet = func.nullif(
        func.ts_headline(lang, DBMeta.inhalt, tsq, type_=sa.Text()),
        "",
    )
    if not include_metadata:
        return content_snippet.label("snippet")
    meta_text = _metadata_text()
    return func.coalesce(
        content_snippet,
        func.ts_headline(lang, meta_text, tsq, type_=sa.Text()),
    ).label("snippet")


def search_documents(
    session: Session,
    *,
    pool_id: int,
    query: str,
    lang: str = "german",
    limit: int = 50,
    mode: QueryMode = "websearch",
    sort_by: SortField = "sort_date",
    sort_desc: bool = True,
    include_metadata: bool = True,
    sort_date_from: datetime | None = None,
    sort_date_to: datetime | None = None,
    suffixes: Sequence[str] | None = None,
    path_prefix: str | None = None,
) -> DocSearchResult:
    """Search or list indexed documents.

    With a non-empty *query*, matches extracted content (``ts_content`` / ``inhalt``)
    and optionally metadata (filename, path, content type, and Tika ``meta_data`` JSON).
    With an empty *query*, returns all rows that satisfy the filter conditions.
    All matching rows are counted; only up to ``limit`` hits are returned.
    """
    text = query.strip()
    tsq = _tsquery(lang, text, mode) if text else None
    meta_ts = _metadata_tsvector(lang) if text else None
    where = _search_conditions(
        pool_id=pool_id,
        tsq=tsq,
        meta_ts=meta_ts,
        include_metadata=include_metadata,
        sort_date_from=sort_date_from,
        sort_date_to=sort_date_to,
        suffixes=suffixes,
        path_prefix=path_prefix,
    )

    total_count = session.scalar(
        select(func.count()).select_from(DBMeta).where(*where)
    )
    total_count = int(total_count or 0)
    if total_count == 0:
        return DocSearchResult(hits=[], total_count=0)

    snippet = (
        _snippet_expr(lang, tsq, include_metadata=include_metadata)
        if tsq is not None
        else sa.literal("").label("snippet")
    )

    order_primary, order_secondary = _order_by(sort_by, sort_desc)
    stmt = (
        select(
            DBMeta.id.label("meta_id"),
            DBMeta.path,
            DBMeta.fname,
            DBMeta.suffix,
            DBMeta.sort_date,
            DBMeta.thumbarray,
            snippet,
        )
        .where(*where)
        .order_by(order_primary, order_secondary)
        .limit(limit)
    )

    hits = [
        DocSearchHit(
            meta_id=row.meta_id,
            path=row.path,
            fname=row.fname,
            suffix=row.suffix,
            sort_date=row.sort_date,
            snippet=row.snippet or "",
            thumbarray=row.thumbarray,
        )
        for row in session.execute(stmt)
    ]
    return DocSearchResult(hits=hits, total_count=total_count)
