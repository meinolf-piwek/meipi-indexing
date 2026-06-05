"""Tests for configuration helpers."""

import os

import pytest

from meipi.indexing.config import Config


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".PDF", "doc"),
        (".jpg", "pic"),
        (".HEIC", "pic"),
        (".mp4", "vid"),
        (".xyz", "unk"),
    ],
)
def test_get_ftype(test_config: Config, suffix: str, expected: str):
    assert test_config.get_ftype(suffix) == expected


def test_db_conn_url_uses_config_and_port(test_config: Config):
    url = test_config.db_conn_URL
    assert url.drivername == "postgresql+psycopg"
    assert url.username == "testuser"
    assert url.password == "test-secret"
    assert url.host == "localhost"
    assert url.port == 5432
    assert url.database == "testdb"


def test_db_passwd_skips_keyring_without_dbus(
    test_config: Config, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    assert test_config.db_passwd_from_keyring() == "test-secret"


def test_resolved_docroot(test_config: Config):
    assert test_config.resolved_docroot() == os.path.abspath(test_config.docroot)


def test_db_operations_sets_search_path(test_config: Config, sample_pool, monkeypatch):
    from meipi.indexing.operations import DBOperations

    captured: dict = {}

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("meipi.indexing.operations.sa.create_engine", fake_create_engine)
    monkeypatch.setattr("meipi.indexing.operations.sessionmaker", lambda **kw: lambda: None)

    DBOperations(pool=sample_pool, config=test_config)

    assert captured["kwargs"]["connect_args"] == {
        "options": f"-c search_path={test_config.pg_schema}"
    }
