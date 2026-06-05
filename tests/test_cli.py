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


def test_check_sync_subcommand_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["check-sync", "--help"])
    assert result.exit_code == 0
    assert "--pool-id" in result.output
    assert "--json" in result.output
    assert "--verbose" in result.output
    assert "RELPATH" in result.output


def test_check_sync_invokes_check_pool_sync(monkeypatch):
    from meipi.indexing.watcher import SyncReport

    report = SyncReport(
        schema="public",
        pool_id=2,
        docroot="/data",
        watch_relpath=".",
        schema_tables={"filemeta": 1},
        tables_missing=(),
        filemeta_rows=1,
        pool_indexed_count=1,
        fs_count=1,
        watch_indexed_count=1,
        in_sync=("a.txt",),
        only_on_disk=(),
        only_in_db=(),
    )
    captured: dict[str, Any] = {}

    def fake_check(dbop, *, watch_relpath):
        captured["watch_relpath"] = watch_relpath
        return report

    monkeypatch.setattr("meipi.indexing.cmd.cli.check_pool_sync", fake_check)
    monkeypatch.setattr(
        "meipi.indexing.cmd.cli.DBOperations",
        lambda pool_id: type("FakeDB", (), {"pool_id": pool_id})(),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["check-sync", "--pool-id", "2", "."])
    assert result.exit_code == 0
    assert captured["watch_relpath"] == "."
    assert "Schema: public" in result.output
    assert "Filesystem and database are in sync" in result.output
    assert "filemeta: 1 rows in schema" in result.output


def test_check_sync_verbose_lists_paths(monkeypatch):
    from meipi.indexing.watcher import SyncReport

    report = SyncReport(
        schema="public",
        pool_id=1,
        docroot="/data",
        watch_relpath=".",
        schema_tables={"filemeta": 3},
        tables_missing=(),
        filemeta_rows=3,
        pool_indexed_count=3,
        fs_count=2,
        watch_indexed_count=1,
        in_sync=("keep.txt",),
        only_on_disk=("new.txt",),
        only_in_db=(),
    )

    monkeypatch.setattr(
        "meipi.indexing.cmd.cli.check_pool_sync",
        lambda dbop, *, watch_relpath: report,
    )
    monkeypatch.setattr(
        "meipi.indexing.cmd.cli.DBOperations",
        lambda pool_id: object(),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["check-sync", "--pool-id", "1", "-v"])
    assert result.exit_code == 0
    assert "  - new.txt" in result.output
    assert "filemeta" in result.output


def test_watch_subcommand_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--pool-id" in result.output
    assert "--no-startup-check" in result.output
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
