"""Move thumbarray from pictures to filemeta.

Revision ID: 20260606a
Revises: 20260603a
Create Date: 2026-06-06

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260606a"
down_revision: Union[str, Sequence[str], None] = "20260603a"
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
    op.add_column(
        "filemeta",
        sa.Column("thumbarray", sa.LargeBinary(), nullable=True),
        schema=schema,
    )
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}"."filemeta" AS fm
            SET thumbarray = p.thumbarray
            FROM "{schema}"."pictures" AS p
            WHERE p.meta_id = fm.id
            """
        )
    )
    op.drop_column("pictures", "thumbarray", schema=schema)


def downgrade() -> None:
    schema = _schema()
    op.add_column(
        "pictures",
        sa.Column("thumbarray", sa.LargeBinary(), nullable=True),
        schema=schema,
    )
    op.execute(
        sa.text(
            f"""
            UPDATE "{schema}"."pictures" AS p
            SET thumbarray = fm.thumbarray
            FROM "{schema}"."filemeta" AS fm
            WHERE p.meta_id = fm.id
            """
        )
    )
    op.drop_column("filemeta", "thumbarray", schema=schema)
