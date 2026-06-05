"""Tests for mutable appconf and Config."""

import pytest
from sqlalchemy import MetaData

from meipi.indexing import appconf, reload_appconf
from meipi.indexing.config import Config
from meipi.indexing.model import Base, DBMeta, orm_metadata


def test_appconf_is_mutable():
    original = appconf.docroot
    try:
        appconf.docroot = "/tmp/mutable-docroot"
        assert appconf.docroot == "/tmp/mutable-docroot"
    finally:
        appconf.docroot = original


def test_config_metadata_is_shared_orm_registry():
    cfg = Config.load("config.env")
    assert cfg.metadata is orm_metadata
    assert cfg.metadata is Base.metadata
    assert isinstance(cfg.metadata, MetaData)
    assert cfg.metadata.schema is None


def test_orm_tables_are_unqualified():
    assert DBMeta.__table__.schema is None


def test_db_operations_search_path_follows_pg_schema(
    test_config, sample_pool, monkeypatch
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


def test_reload_appconf_updates_appconf_in_place(tmp_path):
    import meipi.indexing

    env = tmp_path / "reload.env"
    env.write_text(
        "IND_DOCROOT=/tmp/reloaded-docroot\nIND_PG_SCHEMA=reload_schema\n",
        encoding="utf-8",
    )
    current = meipi.indexing.appconf
    original = current.model_copy()
    try:
        reloaded = reload_appconf(str(env))
        assert reloaded is current
        assert meipi.indexing.appconf is current
        assert appconf.docroot == "/tmp/reloaded-docroot"
        assert appconf.pg_schema == "reload_schema"
        assert appconf.envfile == str(env)
    finally:
        for field_name in Config.model_fields:
            setattr(current, field_name, getattr(original, field_name))


def test_reload_appconf_uses_meipi_config_env(tmp_path, monkeypatch):
    import meipi.indexing

    env = tmp_path / "from-env-var.env"
    env.write_text("IND_DOCROOT=/tmp/from-env-var\n", encoding="utf-8")
    monkeypatch.setenv("MEIPI_CONFIG_ENV", str(env))
    current = meipi.indexing.appconf
    original = current.model_copy()
    try:
        reload_appconf()
        assert appconf.docroot == "/tmp/from-env-var"
        assert appconf.envfile == str(env)
    finally:
        for field_name in Config.model_fields:
            setattr(current, field_name, getattr(original, field_name))
        monkeypatch.delenv("MEIPI_CONFIG_ENV", raising=False)
