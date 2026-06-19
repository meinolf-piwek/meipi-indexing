"""Point bge_m3_vectors.doc_id at filemeta and drop documents.

Revision ID: 20260618a
Revises: 20260612a
Create Date: 2026-06-18

Data migration:
- ``bge_m3_vectors.doc_id`` previously referenced ``documents.id``
- it now references ``filemeta.id`` (the indexed file row)

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260618a"
down_revision: Union[str, Sequence[str], None] = "20260612a"
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
    documents = f'"{schema}"."documents"'
    filemeta = f'"{schema}"."filemeta"'

    # Remove chunk rows that cannot be mapped to a filemeta row.
    op.execute(
        sa.text(
            f"DELETE FROM {vectors} AS v "
            f"WHERE NOT EXISTS ("
            f"  SELECT 1 FROM {documents} AS d WHERE d.id = v.doc_id"
            f") AND NOT EXISTS ("
            f"  SELECT 1 FROM {filemeta} AS f WHERE f.id = v.doc_id"
            f")"
        )
    )

    # Drop the old FK before rewriting doc_id values to filemeta ids.
    op.drop_constraint(
        "bge_m3_vectors_doc_id_fkey",
        "bge_m3_vectors",
        schema=schema,
        type_="foreignkey",
    )

    # Remap legacy rows: documents.id -> filemeta.id via documents.meta_id.
    # Rows already keyed by filemeta.id are left unchanged.
    op.execute(
        sa.text(
            f"UPDATE {vectors} AS v "
            f"SET doc_id = d.meta_id "
            f"FROM {documents} AS d "
            "WHERE v.doc_id = d.id"
        )
    )

    # Drop rows that still do not reference filemeta after remapping.
    op.execute(
        sa.text(
            f"DELETE FROM {vectors} AS v "
            f"WHERE NOT EXISTS ("
            f"  SELECT 1 FROM {filemeta} AS f WHERE f.id = v.doc_id"
            f")"
        )
    )

    op.create_foreign_key(
        "bge_m3_vectors_doc_id_fkey",
        "bge_m3_vectors",
        "filemeta",
        ["doc_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="CASCADE",
    )

    op.drop_table("documents", schema=schema)


def downgrade() -> None:
    schema = _schema()
    op.execute(f'SET search_path TO "{schema}"')

    vectors = f'"{schema}"."bge_m3_vectors"'
    documents = f'"{schema}"."documents"'
    filemeta = f'"{schema}"."filemeta"'

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("meta_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["meta_id"],
            [f"{schema}.filemeta.id"],
            name="documents_meta_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="documents_pkey"),
        sa.UniqueConstraint("meta_id", name="documents_meta_id_key"),
        schema=schema,
    )

    # One documents row per filemeta referenced by chunk vectors.
    op.execute(
        sa.text(
            f"INSERT INTO {documents} (meta_id) "
            f"SELECT DISTINCT v.doc_id "
            f"FROM {vectors} AS v "
            f"JOIN {filemeta} AS f ON f.id = v.doc_id "
            "ORDER BY v.doc_id"
        )
    )

    op.drop_constraint(
        "bge_m3_vectors_doc_id_fkey",
        "bge_m3_vectors",
        schema=schema,
        type_="foreignkey",
    )

    op.execute(
        sa.text(
            f"UPDATE {vectors} AS v "
            f"SET doc_id = d.id "
            f"FROM {documents} AS d "
            "WHERE d.meta_id = v.doc_id"
        )
    )

    op.create_foreign_key(
        "bge_m3_vectors_doc_id_fkey",
        "bge_m3_vectors",
        "documents",
        ["doc_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="CASCADE",
    )
