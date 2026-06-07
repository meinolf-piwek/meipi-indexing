"""Fixtures for PostgreSQL integration tests.

Set ``MEIPI_TEST_DATABASE_URL`` to a dedicated test database, for example::

    export MEIPI_TEST_DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/meipi_indexing_test"

The database is wiped per test (``recreate_tables``). Do not point this at production data.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from meipi.indexing.config import Config
from meipi.indexing.model import DBMeta, DBPic, DBPool
from meipi.indexing.operations import DBOperations

pytestmark = pytest.mark.integration

_ENV_URL = "MEIPI_TEST_DATABASE_URL"


def _integration_config() -> Config:
    raw = os.environ.get(_ENV_URL)
    if not raw:
        pytest.skip(
            f"{_ENV_URL} is not set — skipping PostgreSQL integration tests"
        )
    url = make_url(raw)
    if url.drivername not in {"postgresql", "postgresql+psycopg", "postgresql+psycopg2"}:
        pytest.skip(f"{_ENV_URL} must use a PostgreSQL driver, got {url.drivername!r}")
    return Config(
        pg_host=url.host or "localhost",
        pg_port=str(url.port or 5432),
        pg_user=url.username or "postgres",
        pg_passwd=url.password or "",
        pg_database=url.database or "postgres",
        pg_schema="public",
    )


@pytest.fixture(scope="session")
def integration_config():
    """Config from MEIPI_TEST_DATABASE_URL without using the host keyring."""
    config = _integration_config()

    def _password_from_config(self: Config) -> str:
        return self.pg_passwd

    original = Config.db_passwd_from_keyring
    Config.db_passwd_from_keyring = _password_from_config  # type: ignore[method-assign]
    yield config
    Config.db_passwd_from_keyring = original  # type: ignore[method-assign]


@pytest.fixture(scope="session")
def pg_engine(integration_config: Config):
    connect_args = {"options": f"-c search_path={integration_config.pg_schema}"}
    engine = sa.create_engine(
        integration_config.db_conn_URL,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable ({_ENV_URL}): {exc}")

    try:
        with engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:
        pytest.skip(f"pgvector extension not available: {exc}")

    yield engine
    engine.dispose()


@pytest.fixture
def db_ops(pg_engine, integration_config: Config, tmp_path):
    """Fresh schema and DBOperations bound to a persisted datapool row."""
    docroot = tmp_path / "files"
    docroot.mkdir()
    integration_config_with_root = integration_config.model_copy(
        update={"docroot": str(docroot)}
    )

    ops = DBOperations(
        pool=DBPool(
            id=0,
            pool="integration-pool",
            description="integration test",
        ),
        config=integration_config_with_root,
    )
    ops.engine.dispose()
    ops.engine = pg_engine
    ops.Session = sessionmaker(bind=pg_engine, expire_on_commit=False)

    ops.recreate_tables()

    pool_name = f"integration-{tmp_path.name}"
    with ops.Session() as session:
        session.execute(sa.delete(DBPool))
        session.commit()
        pool_row = DBPool(
            id=1,
            pool=pool_name,
            description="integration test",
        )
        session.add(pool_row)
        session.commit()
        session.refresh(pool_row)
        ops.pool = pool_row

    yield ops

    ops.recreate_tables()


def seed_pic_meta(
    ops: DBOperations,
    *,
    path: str,
    suffix: str,
    with_thumb: bool = False,
) -> tuple[DBMeta, DBPic]:
    """Insert filemeta + pictures rows for integration tests."""
    now = datetime(2024, 1, 15, 10, 0, 0)
    with ops.Session() as session:
        meta = DBMeta(
            pool_id=ops.pool.id,
            path=path,
            fname=path.rsplit("/", maxsplit=1)[-1],
            suffix=suffix,
            sort_date=now,
            fdate=now,
            fsize=50,
            clength=50,
            ctype="image/jpeg",
            ftype="pic",
            md_keys=[],
            meta_data={},
            sha256=b"\xab" * 32,
        )
        meta.thumbarray = (
            np.zeros((4, 4, 3), dtype=np.uint8) if with_thumb else None
        )
        meta.pic = DBPic(
            xmp={},
            truncated=False,
        )
        session.add(meta)
        session.commit()
        session.refresh(meta)
        session.refresh(meta.pic)
        pic = meta.pic
        assert pic is not None

    return meta, pic
