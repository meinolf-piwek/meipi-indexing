"""Drop chunk_id; use composite primary key (doc_id, chunk_index).

Revision ID: 20260619a
Revises: 20260618a
Create Date: 2026-06-19

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260619a"
down_revision: Union[str, Sequence[str], None] = "20260618a"
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
    op.execute(f'SET search_path TO "{schema}"')

    vectors = f'"{schema}"."bge_m3_vectors"'

    # Keep one row per (doc_id, chunk_index) before changing the primary key.
    op.execute(
        sa.text(
            f"DELETE FROM {vectors} AS newer "
            f"USING {vectors} AS older "
            "WHERE newer.doc_id = older.doc_id "
            "AND newer.chunk_index = older.chunk_index "
            "AND newer.chunk_id > older.chunk_id"
        )
    )

    op.drop_constraint(
        "bge_m3_vectors_pkey",
        "bge_m3_vectors",
        schema=schema,
        type_="primary",
    )
    op.drop_constraint(
        "uq_bge_m3_vectors_doc_chunk",
        "bge_m3_vectors",
        schema=schema,
        type_="unique",
    )
    op.drop_column("bge_m3_vectors", "chunk_id", schema=schema)
    op.create_primary_key(
        "bge_m3_vectors_pkey",
        "bge_m3_vectors",
        ["doc_id", "chunk_index"],
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.execute(f'SET search_path TO "{schema}"')

    vectors = f'"{schema}"."bge_m3_vectors"'

    op.drop_constraint(
        "bge_m3_vectors_pkey",
        "bge_m3_vectors",
        schema=schema,
        type_="primary",
    )
    op.add_column(
        "bge_m3_vectors",
        sa.Column("chunk_id", sa.Integer(), nullable=True),
        schema=schema,
    )
    op.execute(
        sa.text(
            f"WITH numbered AS ("
            f"  SELECT doc_id, chunk_index, "
            f"         row_number() OVER (ORDER BY doc_id, chunk_index) AS new_chunk_id "
            f"  FROM {vectors}"
            f") "
            f"UPDATE {vectors} AS v "
            f"SET chunk_id = numbered.new_chunk_id "
            f"FROM numbered "
            f"WHERE v.doc_id = numbered.doc_id "
            f"AND v.chunk_index = numbered.chunk_index"
        )
    )
    op.alter_column(
        "bge_m3_vectors",
        "chunk_id",
        nullable=False,
        schema=schema,
    )
    op.create_primary_key(
        "bge_m3_vectors_pkey",
        "bge_m3_vectors",
        ["chunk_id"],
        schema=schema,
    )
    op.create_unique_constraint(
        "uq_bge_m3_vectors_doc_chunk",
        "bge_m3_vectors",
        ["doc_id", "chunk_index"],
        schema=schema,
    )
    op.execute(
        sa.text(
            f"CREATE SEQUENCE IF NOT EXISTS \"{schema}\".bge_m3_vectors_chunk_id_seq "
            f"OWNED BY {vectors}.chunk_id"
        )
    )
    op.execute(
        sa.text(
            f"SELECT setval("
            f"  '\"{schema}\".bge_m3_vectors_chunk_id_seq', "
            f"  COALESCE((SELECT MAX(chunk_id) FROM {vectors}), 1)"
            f")"
        )
    )
    op.alter_column(
        "bge_m3_vectors",
        "chunk_id",
        server_default=sa.text(f"nextval('\"{schema}\".bge_m3_vectors_chunk_id_seq'::regclass)"),
        schema=schema,
    )
