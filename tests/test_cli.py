"""Tests for the meipi-index CLI."""

from typing import Any

from click.testing import CliRunner

from meipi.indexing.cmd.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Index documents and images" in result.output
    assert "--docroot" in result.output


def test_read_files_subcommand_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["read-files", "--help"])
    assert result.exit_code == 0
    assert "--pool-id" in result.output
    assert "RELPATH" in result.output


def test_watch_subcommand_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--pool-id" in result.output
    assert "--initial-scan" in result.output
    assert "RELPATH" in result.output


def test_schema_info_invokes_db_operations(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeDBOperations:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def schema_info(self):
            return {"schema": "public", "tables": {"filemeta": 42}}

    monkeypatch.setattr("meipi.indexing.cmd.cli.DBOperations", FakeDBOperations)

    runner = CliRunner()
    result = runner.invoke(cli, ["schema-info"])
    assert result.exit_code == 0
    assert captured["kwargs"]["allow_no_pool"] is True
    assert "Schema: public" in result.output
    assert "filemeta" in result.output
    assert "42" in result.output


def test_read_files_sets_docroot_on_appconf(monkeypatch, tmp_path):
    async def noop_bulk(**_kwargs):
        return None

    monkeypatch.setattr("meipi.indexing.cmd.cli.read_files_bulk", noop_bulk)
    monkeypatch.setattr(
        "meipi.indexing.cmd.cli.DBOperations",
        lambda pool_id: type("FakeDB", (), {"pool": type("P", (), {"id": pool_id})()})(),
    )

    docroot = tmp_path / "data"
    docroot.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--docroot",
            str(docroot),
            "read-files",
            "--pool-id",
            "1",
            ".",
            "--no-thumbs",
        ],
    )
    assert result.exit_code == 0

    import meipi.indexing

    assert meipi.indexing.appconf.docroot == str(docroot)
