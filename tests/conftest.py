"""Shared fixtures for meipi-indexing tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

from meipi.indexing.config import Config
from meipi.indexing.model import DBPool, DBServerPool


@pytest.fixture
def test_docroot(tmp_path):
    docroot = tmp_path / "docs"
    docroot.mkdir()
    return docroot


@pytest.fixture
def test_config(tmp_path, monkeypatch: pytest.MonkeyPatch, test_docroot) -> Config:
    """Config that never touches the system keyring."""
    monkeypatch.setattr(
        Config,
        "db_passwd_from_keyring",
        lambda self: "test-secret",
    )
    return Config(
        pg_host="localhost",
        pg_port="5432",
        pg_user="testuser",
        pg_passwd="test-secret",
        pg_database="testdb",
        pg_schema="public",
        server_name="test-server",
    )


@pytest.fixture
def sample_pool() -> DBPool:
    return DBPool(
        id=1,
        pool="test-pool",
        description="fixture pool",
    )


@pytest.fixture
def sample_serverpool(sample_pool: DBPool, test_docroot) -> DBServerPool:
    return DBServerPool(
        server_name="test-server",
        pool_id=sample_pool.id,
        docroot=str(test_docroot),
    )


@pytest.fixture
def patch_serverpool_lookup(monkeypatch: pytest.MonkeyPatch, sample_serverpool: DBServerPool):
    monkeypatch.setattr(
        "meipi.indexing.operations.DBOperations.get_serverpool",
        lambda self: sample_serverpool,
    )


@pytest.fixture
def mock_db_operations(
    test_config: Config,
    sample_pool: DBPool,
    sample_serverpool: DBServerPool,
    test_docroot,
    monkeypatch: pytest.MonkeyPatch,
):
    """DBOperations with a mocked engine and controllable session."""
    from meipi.indexing.operations import DBOperations

    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False

    engine = MagicMock()
    SessionFactory = MagicMock(return_value=session)

    monkeypatch.setattr(
        DBOperations,
        "__init__",
        lambda self, pool_id=None, pool=None, *, allow_no_pool=False, config=test_config, enginekwargs=None, sessionkwargs=None: None,
    )

    ops = DBOperations.__new__(DBOperations)
    ops.config = test_config
    ops.logger = test_config.logger
    ops.pool = sample_pool
    ops.pool_id = sample_pool.id
    ops.server_name = test_config.server_name
    ops.serverpool = sample_serverpool
    ops.docroot = str(test_docroot)
    ops.engine = engine
    ops.Session = SessionFactory
    ops.metadata = sa.MetaData()
    return ops, session
