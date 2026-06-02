"""Shared helpers for unit and integration tests."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.dialects import postgresql

from meipi.indexing.model import DBMeta

POSTGRES_DIALECT = postgresql.dialect()


def compile_postgres(statement) -> str:
    """Compile a SQLAlchemy statement to PostgreSQL SQL text."""
    return str(statement.compile(dialect=POSTGRES_DIALECT, compile_kwargs={"literal_binds": True}))


def make_dbmeta(
    *,
    meta_id: int | None = 42,
    pool_id: int = 1,
    path: str = "subdir/file.pdf",
    ftype: str = "doc",
    suffix: str = ".pdf",
    **kwargs,
) -> DBMeta:
    """Build a DBMeta instance with sensible defaults for unit tests."""
    now = datetime(2024, 6, 1, 12, 0, 0)
    defaults = dict(
        pool_id=pool_id,
        path=path,
        fname=kwargs.pop("fname", path.rsplit("/", maxsplit=1)[-1]),
        suffix=suffix,
        sort_date=now,
        fdate=now,
        fsize=100,
        clength=100,
        ctype="application/pdf",
        ftype=ftype,
        md_keys=["Content-Type"],
        meta_data={"Content-Type": "application/pdf"},
        sha256=b"\x00" * 32,
        doc=None,
        pic=None,
        vid=None,
    )
    defaults.update(kwargs)
    meta = DBMeta(**defaults)
    if meta_id is not None:
        object.__setattr__(meta, "id", meta_id)
    return meta
