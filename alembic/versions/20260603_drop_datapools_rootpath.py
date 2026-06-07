"""Remove rootpath from datapools; filesystem root is configured via IND_DOCROOT.

Revision ID: 20260603a
Revises: 20260529a
Create Date: 2026-06-03

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260603a"
down_revision: Union[str, Sequence[str], None] = "20260529a"
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
    op.drop_column("datapools", "rootpath", schema=schema)


def downgrade() -> None:
    schema = _schema()
    op.add_column(
        "datapools",
        sa.Column("rootpath", sa.String(), nullable=False, server_default="."),
        schema=schema,
    )
    op.alter_column("datapools", "rootpath", server_default=None, schema=schema)
