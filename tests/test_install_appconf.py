"""Tests for install_appconf."""

import inspect

import meipi.indexing
from meipi.indexing.config import install_appconf
from meipi.indexing.operations import DBOperations


def test_install_appconf_updates_module_bindings():
    original = meipi.indexing.appconf
    try:
        updated = original.model_copy(update={"docroot": "/tmp/override"})
        install_appconf(updated)
        assert meipi.indexing.appconf.docroot == "/tmp/override"
    finally:
        install_appconf(original)


def test_dboperations_uses_appconf_when_config_default_none():
    original = meipi.indexing.appconf.docroot
    try:
        meipi.indexing.appconf.docroot = "/tmp/dbops-docroot"
        sig = inspect.signature(DBOperations.__init__)
        assert sig.parameters["config"].default is None
        # Cannot construct without DB; resolved docroot is read at __init__ time
        assert meipi.indexing.appconf.resolved_docroot() == "/tmp/dbops-docroot"
    finally:
        meipi.indexing.appconf.docroot = original
