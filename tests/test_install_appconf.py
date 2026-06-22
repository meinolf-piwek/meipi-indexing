"""Tests for install_appconf."""

import inspect

import meipi.indexing
from meipi.indexing.config import install_appconf
from meipi.indexing.operations import DBOperations


def test_install_appconf_updates_module_bindings():
    original = meipi.indexing.appconf
    try:
        updated = original.model_copy(update={"server_name": "override-server"})
        install_appconf(updated)
        assert meipi.indexing.appconf.server_name == "override-server"
    finally:
        install_appconf(original)


def test_dboperations_uses_appconf_when_config_default_none():
    original = meipi.indexing.appconf.server_name
    try:
        meipi.indexing.appconf.server_name = "dbops-server"
        sig = inspect.signature(DBOperations.__init__)
        assert sig.parameters["config"].default is None
        assert meipi.indexing.appconf.server_name == "dbops-server"
    finally:
        meipi.indexing.appconf.server_name = original
