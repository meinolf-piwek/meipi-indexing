"""Add serverpools table mapping server_name and pool to filesystem docroot.

Revision ID: 20260621a
Revises: 20260620a
Create Date: 2026-06-21

Existing datapools rows are seeded with ``server_name`` from config (default
``lenox``) and ``docroot`` from ``IND_DOCROOT`` (default ``.``).
"""

from __future__ import annotations

import os
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260621a"
down_revision: Union[str, Sequence[str], None] = "20260620a"
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


def _server_name() -> str:
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from meipi.indexing.config import CONFIG_PATH, Config

    return Config(_env_file=CONFIG_PATH, config_path=CONFIG_PATH).server_name


def _default_docroot() -> str:
    return os.environ.get("IND_DOCROOT", ".")


def upgrade() -> None:
    schema = _schema()
    op.execute(f'SET search_path TO "{schema}"')

    serverpools = f'"{schema}"."serverpools"'
    datapools = f'"{schema}"."datapools"'

    op.create_table(
        "serverpools",
        sa.Column("server_name", sa.String(), nullable=False),
        sa.Column("pool_id", sa.Integer(), nullable=False),
        sa.Column("docroot", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["pool_id"],
            [f"{schema}.datapools.id"],
            name="serverpools_pool_id_fkey",
        ),
        sa.PrimaryKeyConstraint("server_name", "pool_id", name="serverpools_pkey"),
        schema=schema,
    )

    server_name = _server_name()
    docroot = _default_docroot()
    conn = op.get_bind()
    conn.execute(
        sa.text(
            f"INSERT INTO {serverpools} (server_name, pool_id, docroot) "
            f"SELECT :server_name, id, :docroot FROM {datapools}"
        ),
        {"server_name": server_name, "docroot": docroot},
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_table("serverpools", schema=schema)
