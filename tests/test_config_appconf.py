"""Tests for mutable appconf and Config."""

import pytest
from sqlalchemy import MetaData

from meipi.indexing import appconf
from meipi.indexing.config import Config
from meipi.indexing.model import Base, DBMeta, orm_metadata


def test_appconf_is_mutable():
    original = appconf.server_name
    try:
        appconf.server_name = "other-server"
        assert appconf.server_name == "other-server"
    finally:
        appconf.server_name = original


def test_config_metadata_is_shared_orm_registry():
    cfg = Config(_env_file="config.env", config_path="config.env")
    assert cfg.metadata is orm_metadata
    assert cfg.metadata is Base.metadata
    assert isinstance(cfg.metadata, MetaData)
    assert cfg.metadata.schema is None


def test_orm_tables_are_unqualified():
    assert DBMeta.__table__.schema is None


def test_db_operations_search_path_follows_pg_schema(
    test_config, sample_pool, patch_serverpool_lookup, monkeypatch
):
    from meipi.indexing.operations import DBOperations

    captured: dict = {}

    def fake_create_engine(url, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("meipi.indexing.operations.sa.create_engine", fake_create_engine)
    monkeypatch.setattr("meipi.indexing.operations.sessionmaker", lambda **kw: lambda: None)

    test_config.pg_schema = "custom_schema"
    DBOperations(pool=sample_pool, config=test_config)

    assert captured["kwargs"]["connect_args"] == {
        "options": "-c search_path=custom_schema"
    }


def test_install_appconf_replaces_appconf():
    import meipi.indexing
    from meipi.indexing.config import install_appconf

    original = meipi.indexing.appconf
    try:
        updated = original.model_copy(update={"pg_schema": "install_test_schema"})
        install_appconf(updated)
        assert meipi.indexing.appconf.pg_schema == "install_test_schema"
        assert DBMeta.__table__.schema is None
    finally:
        install_appconf(original)

