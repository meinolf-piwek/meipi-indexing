"""Tests for watcher startup filesystem/database sync check."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from meipi.indexing.operations import DBOperations
from meipi.indexing.paths import is_under_watch_tree, list_watch_files, normalize_rel_path
from meipi.indexing.watcher import check_pool_sync


def test_check_pool_sync_reports_differences_without_applying(
    tmp_path: Path,
    sample_pool,
    test_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docroot = tmp_path / "data"
    docs = docroot / "docs"
    docs.mkdir(parents=True)
    (docs / "keep.txt").write_text("keep", encoding="utf-8")
    (docs / "new.txt").write_text("new", encoding="utf-8")

    dbop = MagicMock(spec=DBOperations)
    dbop.docroot = str(docroot)
    dbop.pool = sample_pool
    dbop.schema_info.return_value = {
        "schema": "public",
        "tables": {"datapools": 1, "filemeta": 10_500},
    }
    dbop.count_pool_filemeta.return_value = 10_234
    dbop.list_pool_file_paths.return_value = {"docs/keep.txt", "docs/stale.txt"}

    report = check_pool_sync(dbop, watch_relpath="docs")

    dbop.ensure_tables.assert_not_called()
    dbop.delete_file_meta.assert_not_called()
    assert report.schema == "public"
    assert report.filemeta_rows == 10_500
    assert report.pool_indexed_count == 10_234
    assert report.watch_indexed_count == 2
    assert report.tables_missing == (
        "pictures",
        "videos",
        "dino_v2_vectors",
        "bge_m3_vectors",
    )
    assert report.in_sync == ("docs/keep.txt",)
    assert report.only_in_db == ("docs/stale.txt",)
    assert report.only_on_disk == ("docs/new.txt",)
    assert report.has_differences is True


def test_is_under_watch_tree_uses_forward_slashes() -> None:
    assert is_under_watch_tree("docs/a.txt", "docs") is True
    assert is_under_watch_tree("docs\\a.txt", "docs") is True


def test_normalize_rel_path_converts_backslashes() -> None:
    assert normalize_rel_path("docs\\a.txt") == "docs/a.txt"


def test_resolve_watch_relpath_relative_to_docroot(tmp_path: Path) -> None:
    from meipi.indexing.paths import ensure_watch_directory, resolve_watch_relpath

    docroot = tmp_path / "data"
    docs = docroot / "docs"
    docs.mkdir(parents=True)

    assert resolve_watch_relpath(str(docroot), "docs") == "docs"
    assert resolve_watch_relpath(str(docroot), "./docs") == "docs"
    assert ensure_watch_directory(str(docroot), "docs") == "docs"


def test_list_watch_files_relative_path_not_cwd(tmp_path: Path, monkeypatch) -> None:
    docroot = tmp_path / "data"
    docs = docroot / "docs"
    docs.mkdir(parents=True)
    (docs / "in.txt").write_text("in", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    paths = list_watch_files(str(docroot), "docs")

    assert paths == {"docs/in.txt"}


def test_resolve_watch_relpath_absolute_under_docroot(tmp_path: Path) -> None:
    from meipi.indexing.paths import resolve_watch_relpath

    docroot = tmp_path / "data"
    sub = docroot / "docs"
    sub.mkdir(parents=True)
    assert resolve_watch_relpath(str(docroot), str(sub)) == "docs"
    assert resolve_watch_relpath(str(docroot), str(docroot)) == "."


def test_resolve_watch_relpath_rejects_path_outside_docroot(tmp_path: Path) -> None:
    from meipi.indexing.paths import resolve_watch_relpath

    docroot = tmp_path / "data"
    docroot.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(ValueError, match="not under docroot"):
        resolve_watch_relpath(str(docroot), str(outside))


def test_format_sync_report_warns_on_watch_path_mismatch() -> None:
    from meipi.indexing.watcher import SyncReport, format_sync_report

    report = SyncReport(
        schema="public",
        pool_id=1,
        docroot="/data",
        watch_relpath="/home/user/Dokumente",
        schema_tables={"filemeta": 1000},
        tables_missing=(),
        filemeta_rows=1000,
        pool_indexed_count=500,
        fs_count=0,
        watch_indexed_count=0,
        in_sync=(),
        only_on_disk=(),
        only_in_db=(),
    )
    text = format_sync_report(report)

    assert "matches no files on disk" in text
    assert "Filesystem and database are in sync" not in text


def test_list_watch_files_respects_watch_subtree(tmp_path: Path) -> None:
    docroot = tmp_path / "pool"
    docs = docroot / "docs"
    other = docroot / "other"
    docs.mkdir(parents=True)
    other.mkdir()
    (docs / "in.txt").write_text("in", encoding="utf-8")
    (other / "out.txt").write_text("out", encoding="utf-8")

    paths = list_watch_files(str(docroot), "docs")

    assert paths == {"docs/in.txt"}


def test_list_pool_file_paths_filters_by_watch_tree(
    sample_pool, test_config, patch_serverpool_lookup, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    session.scalars.return_value.all.return_value = ["docs/a.txt"]

    monkeypatch.setattr(
        "meipi.indexing.operations.sessionmaker",
        lambda **kwargs: lambda: session,
    )
    monkeypatch.setattr(
        "meipi.indexing.operations.sa.create_engine",
        lambda *args, **kwargs: object(),
    )

    dbop = DBOperations(pool=sample_pool, config=test_config)
    paths = dbop.list_pool_file_paths("docs")

    assert paths == {"docs/a.txt"}


def test_count_pool_filemeta(
    sample_pool, test_config, patch_serverpool_lookup, monkeypatch: pytest.MonkeyPatch
):
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    session.scalar.return_value = 12_345

    monkeypatch.setattr(
        "meipi.indexing.operations.sessionmaker",
        lambda **kwargs: lambda: session,
    )
    monkeypatch.setattr(
        "meipi.indexing.operations.sa.create_engine",
        lambda *args, **kwargs: object(),
    )

    dbop = DBOperations(pool=sample_pool, config=test_config)
    assert dbop.count_pool_filemeta() == 12_345


def test_ensure_tables_creates_only_missing(
    sample_pool, test_config, patch_serverpool_lookup, monkeypatch
):
    dbop = DBOperations(pool=sample_pool, config=test_config)
    created: list = []

    monkeypatch.setattr(
        dbop,
        "schema_info",
        lambda: {"schema": "public", "tables": {"datapools": 1, "filemeta": 2}},
    )
    monkeypatch.setattr(
        dbop,
        "create_tables",
        lambda entities=None: created.append(entities),
    )

    names = dbop.ensure_tables()

    assert names == ["pictures", "videos", "dino_v2_vectors", "bge_m3_vectors"]
    assert len(created) == 1
    assert {entity.__tablename__ for entity in created[0]} == {
        "pictures",
        "videos",
        "dino_v2_vectors",
        "bge_m3_vectors",
    }


def test_sync_report_is_in_sync() -> None:
    from meipi.indexing.watcher import SyncReport

    in_sync = SyncReport(
        schema="public",
        pool_id=1,
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
    out_of_sync = SyncReport(
        schema="public",
        pool_id=1,
        docroot="/data",
        watch_relpath=".",
        schema_tables={"filemeta": 1},
        tables_missing=(),
        filemeta_rows=1,
        pool_indexed_count=1,
        fs_count=2,
        watch_indexed_count=1,
        in_sync=(),
        only_on_disk=("new.txt",),
        only_in_db=(),
    )
    mismatch = SyncReport(
        schema="public",
        pool_id=1,
        docroot="/data",
        watch_relpath="wrong",
        schema_tables={"filemeta": 100},
        tables_missing=(),
        filemeta_rows=100,
        pool_indexed_count=50,
        fs_count=0,
        watch_indexed_count=0,
        in_sync=(),
        only_on_disk=(),
        only_in_db=(),
    )

    assert in_sync.is_in_sync is True
    assert out_of_sync.is_in_sync is False
    assert mismatch.is_in_sync is False


def test_format_sync_report_summary() -> None:
    from meipi.indexing.watcher import SyncReport, format_sync_report

    report = SyncReport(
        schema="meipi-indexing",
        pool_id=2,
        docroot="/data",
        watch_relpath=".",
        schema_tables={"filemeta": 10_500, "datapools": 1},
        tables_missing=("videos",),
        filemeta_rows=10_500,
        pool_indexed_count=10_234,
        fs_count=2,
        watch_indexed_count=2,
        in_sync=("a.txt",),
        only_on_disk=("b.txt",),
        only_in_db=("c.txt",),
    )
    text = format_sync_report(report)

    assert "Schema: meipi-indexing (2 tables)" in text
    assert "filemeta: 10,500 rows in schema" in text
    assert "pool 2: 10,234 indexed paths" in text
    assert "Missing tables: 1" in text
    assert "Only in DB: 1" in text
    assert "Only on disk: 1" in text
    assert "2 files on disk" in text
    assert "2 indexed in tree" in text
    assert "1 matched" in text
    assert "Differences found (no changes applied)" in text
    assert "  - c.txt" not in text


def test_format_sync_report_verbose_lists_details() -> None:
    from meipi.indexing.watcher import SyncReport, format_sync_report

    report = SyncReport(
        schema="meipi-indexing",
        pool_id=1,
        docroot="/data",
        watch_relpath=".",
        schema_tables={"filemeta": 42},
        tables_missing=("videos",),
        filemeta_rows=42,
        pool_indexed_count=40,
        fs_count=1,
        watch_indexed_count=1,
        in_sync=(),
        only_on_disk=("b.txt",),
        only_in_db=("c.txt",),
    )
    text = format_sync_report(report, verbose=True)

    assert "filemeta" in text and "42" in text
    assert "Missing: videos" in text
    assert "  - c.txt" in text
    assert "  - b.txt" in text


def test_sync_report_as_dict_summary_and_verbose() -> None:
    from meipi.indexing.watcher import SyncReport

    report = SyncReport(
        schema="public",
        pool_id=1,
        docroot="/data",
        watch_relpath=".",
        schema_tables={"filemeta": 0},
        tables_missing=(),
        filemeta_rows=0,
        pool_indexed_count=0,
        fs_count=0,
        watch_indexed_count=0,
        in_sync=(),
        only_on_disk=(),
        only_in_db=(),
    )
    summary = report.as_dict()
    verbose = report.as_dict(verbose=True)

    assert summary["schema"] == "public"
    assert summary["filemeta_rows"] == 0
    assert summary["pool_indexed_count"] == 0
    assert summary["has_differences"] is False
    assert summary["matched_count"] == 0
    assert "tables" not in summary
    assert verbose["tables"] == {"filemeta": 0}
