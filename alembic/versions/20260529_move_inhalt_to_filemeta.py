"""Move inhalt and ts_content from documents to filemeta.

Revision ID: 20260529a
Revises:
Create Date: 2026-05-29

"""

from __future__ import annotations

import os
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260529a"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SEARCH_LANGUAGE = "german"


def _schema() -> str:
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from meipi.indexing.config import Config

    env_file = os.environ.get("MEIPI_CONFIG_ENV", "config.env")
    return Config.load(env_file).pg_schema


def upgrade() -> None:
    schema = _schema()
    filemeta = f'"{schema}"."filemeta"'
    documents = f'"{schema}"."documents"'

    op.execute(
        sa.text(
            f"ALTER TABLE {filemeta} "
            "ADD COLUMN IF NOT EXISTS inhalt TEXT NOT NULL DEFAULT ''"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {filemeta} "
            f"ADD COLUMN IF NOT EXISTS ts_content TSVECTOR "
            f"GENERATED ALWAYS AS (to_tsvector('{_SEARCH_LANGUAGE}', inhalt)) STORED"
        )
    )
    op.execute(
        sa.text(
            f"UPDATE {filemeta} AS f "
            f"SET inhalt = d.inhalt "
            f"FROM {documents} AS d "
            "WHERE d.meta_id = f.id AND d.inhalt IS NOT NULL AND d.inhalt <> ''"
        )
    )
    op.execute(
        sa.text(
            f'CREATE INDEX IF NOT EXISTS ix_filemeta_ts_content '
            f"ON {filemeta} USING gin (ts_content)"
        )
    )

    op.execute(sa.text(f"ALTER TABLE {documents} DROP COLUMN IF EXISTS ts_content"))
    op.execute(sa.text(f"ALTER TABLE {documents} DROP COLUMN IF EXISTS inhalt"))


def downgrade() -> None:
    schema = _schema()
    filemeta = f'"{schema}"."filemeta"'
    documents = f'"{schema}"."documents"'

    op.execute(
        sa.text(
            f"ALTER TABLE {documents} "
            "ADD COLUMN IF NOT EXISTS inhalt TEXT NOT NULL DEFAULT ''"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {documents} "
            f"ADD COLUMN IF NOT EXISTS ts_content TSVECTOR "
            f"GENERATED ALWAYS AS (to_tsvector('{_SEARCH_LANGUAGE}', inhalt)) STORED"
        )
    )
    op.execute(
        sa.text(
            f"UPDATE {documents} AS d "
            f"SET inhalt = f.inhalt "
            f"FROM {filemeta} AS f "
            "WHERE d.meta_id = f.id"
        )
    )

    op.execute(sa.text(f"DROP INDEX IF EXISTS {schema}.ix_filemeta_ts_content"))
    op.execute(sa.text(f"ALTER TABLE {filemeta} DROP COLUMN IF EXISTS ts_content"))
    op.execute(sa.text(f"ALTER TABLE {filemeta} DROP COLUMN IF EXISTS inhalt"))
