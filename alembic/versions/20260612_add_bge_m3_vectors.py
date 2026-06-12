"""Add bge_m3_vectors table for document chunks and embeddings.

Revision ID: 20260612a
Revises: 20260607a
Create Date: 2026-06-12

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "20260612a"
down_revision: Union[str, Sequence[str], None] = "20260607a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _schema() -> str:
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from meipi.indexing.config import CONFIG_PATH, Config

    return Config(_env_file=CONFIG_PATH, config_path=CONFIG_PATH).pg_schema


def upgrade() -> None:
    schema = _schema()
    op.create_table(
        "bge_m3_vectors",
        sa.Column("chunk_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("vector", Vector(1024), nullable=True),
        sa.ForeignKeyConstraint(
            ["doc_id"],
            [f"{schema}.documents.id"],
            name="bge_m3_vectors_doc_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id", name="bge_m3_vectors_pkey"),
        sa.UniqueConstraint(
            "doc_id",
            "chunk_index",
            name="uq_bge_m3_vectors_doc_chunk",
        ),
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_table("bge_m3_vectors", schema=schema)
