"""Tests for immutable appconf and Config."""

import pytest

from meipi.indexing import appconf, reload_appconf
from meipi.indexing.config import Config


def test_appconf_is_frozen():
    with pytest.raises(Exception):
        appconf.pg_host = "other"


def test_config_load_returns_frozen_instance():
    cfg = Config.load("config.env")
    with pytest.raises(Exception):
        cfg.pg_database = "other"


def test_reload_appconf_raises():
    with pytest.raises(RuntimeError, match="restart"):
        reload_appconf()
