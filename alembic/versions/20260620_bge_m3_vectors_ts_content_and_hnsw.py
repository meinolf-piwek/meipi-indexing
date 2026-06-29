"""Add ts_content and search indexes to bge_m3_vectors.

Revision ID: 20260620a
Revises: 20260619a
Create Date: 2026-06-20

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260620a"
down_revision: Union[str, Sequence[str], None] = "20260619a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SEARCH_LANGUAGE = "german"


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

    op.execute(
        sa.text(
            f"ALTER TABLE {vectors} "
            f"ADD COLUMN IF NOT EXISTS ts_content TSVECTOR "
            f"GENERATED ALWAYS AS (to_tsvector('{_SEARCH_LANGUAGE}', content)) STORED"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_bge_m3_vectors_ts_content "
            f"ON {vectors} USING gin (ts_content)"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_bge_m3_vectors_vector_hnsw "
            f"ON {vectors} USING hnsw (vector vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64)"
        )
    )


def downgrade() -> None:
    schema = _schema()
    op.execute(f'SET search_path TO "{schema}"')

    vectors = f'"{schema}"."bge_m3_vectors"'

    op.execute(sa.text(f'DROP INDEX IF EXISTS "{schema}".ix_bge_m3_vectors_vector_hnsw'))
    op.execute(sa.text(f'DROP INDEX IF EXISTS "{schema}".ix_bge_m3_vectors_ts_content'))
    op.execute(sa.text(f"ALTER TABLE {vectors} DROP COLUMN IF EXISTS ts_content"))
