"""PostgreSQL full-text search for indexed documents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import sqlalchemy as sa
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

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


def _order_by(sort_by: SortField, sort_desc: bool):
    if sort_by == "path":
        primary = DBMeta.path.desc() if sort_desc else DBMeta.path.asc()
        secondary = DBMeta.sort_date.desc()
    else:
        primary = DBMeta.sort_date.desc() if sort_desc else DBMeta.sort_date.asc()
        secondary = DBMeta.path.asc()
    return primary, secondary


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
) -> list[DocSearchHit]:
    """Search file bodies and metadata with PostgreSQL full-text matching.

    Matches rows where the query hits extracted content (``ts_content`` / ``inhalt``)
    or metadata (filename, path, content type, and Tika ``meta_data`` JSON).
    """
    text = query.strip()
    if not text:
        return []

    tsq = _tsquery(lang, text, mode)
    meta_ts = _metadata_tsvector(lang)
    content_match = DBMeta.ts_content.bool_op("@@")(tsq)
    meta_match = meta_ts.bool_op("@@")(tsq)

    meta_text = _metadata_text()
    snippet = func.coalesce(
        func.nullif(func.ts_headline(lang, DBMeta.inhalt, tsq, type_=sa.Text()), ""),
        func.ts_headline(lang, meta_text, tsq, type_=sa.Text()),
    )

    order_primary, order_secondary = _order_by(sort_by, sort_desc)
    stmt = (
        select(
            DBMeta.id.label("meta_id"),
            DBMeta.path,
            DBMeta.fname,
            DBMeta.suffix,
            DBMeta.sort_date,
            snippet.label("snippet"),
        )
        .where(DBMeta.pool_id == pool_id)
        .where(or_(content_match, meta_match))
        .order_by(order_primary, order_secondary)
        .limit(limit)
    )

    return [
        DocSearchHit(
            meta_id=row.meta_id,
            path=row.path,
            fname=row.fname,
            suffix=row.suffix,
            sort_date=row.sort_date,
            snippet=row.snippet or "",
        )
        for row in session.execute(stmt)
    ]
