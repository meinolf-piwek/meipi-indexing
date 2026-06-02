"""PostgreSQL full-text search for indexed documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sqlalchemy as sa
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .model import DBMeta

QueryMode = Literal["plain", "websearch", "phrase"]


@dataclass(frozen=True, slots=True)
class DocSearchHit:
    """One filemeta row matching a full-text query."""

    meta_id: int
    path: str
    fname: str
    suffix: str
    rank: float
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


def search_documents(
    session: Session,
    *,
    pool_id: int,
    query: str,
    lang: str = "german",
    limit: int = 50,
    mode: QueryMode = "websearch",
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

    rank = (
        func.coalesce(func.ts_rank(DBMeta.ts_content, tsq), 0.0)
        + func.coalesce(func.ts_rank(meta_ts, tsq), 0.0)
    )
    meta_text = _metadata_text()
    snippet = func.coalesce(
        func.nullif(func.ts_headline(lang, DBMeta.inhalt, tsq, type_=sa.Text()), ""),
        func.ts_headline(lang, meta_text, tsq, type_=sa.Text()),
    )

    stmt = (
        select(
            DBMeta.id.label("meta_id"),
            DBMeta.path,
            DBMeta.fname,
            DBMeta.suffix,
            rank.label("rank"),
            snippet.label("snippet"),
        )
        .where(DBMeta.pool_id == pool_id)
        .where(or_(content_match, meta_match))
        .order_by(rank.desc(), DBMeta.path)
        .limit(limit)
    )

    return [
        DocSearchHit(
            meta_id=row.meta_id,
            path=row.path,
            fname=row.fname,
            suffix=row.suffix,
            rank=float(row.rank),
            snippet=row.snippet or "",
        )
        for row in session.execute(stmt)
    ]
