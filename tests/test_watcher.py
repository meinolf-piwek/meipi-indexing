"""Tests for filesystem watcher helpers and handler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from meipi.indexing.watcher import (
    PoolIndexHandler,
    PoolWatcher,
    is_under_watch_tree,
    normalize_rel_path,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a/b/c.txt", "a/b/c.txt"),
        ("./a/./b", "a/b"),
        ("a\\b\\c", "a/b/c"),
    ],
)
def test_normalize_rel_path(path: str, expected: str) -> None:
    assert normalize_rel_path(path) == expected


@pytest.mark.parametrize(
    ("rel_path", "watch_relpath", "expected"),
    [
        ("docs/a.txt", ".", True),
        ("docs/a.txt", "docs", True),
        ("docs/sub/a.txt", "docs", True),
        ("other/a.txt", "docs", False),
        ("docs", "docs", True),
        ("docs-extra/a.txt", "docs", False),
    ],
)
def test_is_under_watch_tree(rel_path: str, watch_relpath: str, expected: bool) -> None:
    assert is_under_watch_tree(rel_path, watch_relpath) is expected


def test_handler_created_indexes_file(tmp_path: Path) -> None:
    docroot = tmp_path / "pool"
    watch_dir = docroot / "docs"
    watch_dir.mkdir(parents=True)
    file_path = watch_dir / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    indexed: list[str] = []
    deleted: list[str] = []
    handler = PoolIndexHandler(
        docroot=str(docroot),
        watch_relpath="docs",
        on_index=indexed.append,
        on_delete=deleted.append,
    )

    handler.on_created(MagicMock(is_directory=False, src_path=str(file_path)))

    assert indexed == ["docs/note.txt"]
    assert deleted == []


def test_handler_deleted_removes_entry(tmp_path: Path) -> None:
    docroot = tmp_path / "pool"
    watch_dir = docroot / "docs"
    watch_dir.mkdir(parents=True)
    file_path = watch_dir / "note.txt"

    indexed: list[str] = []
    deleted: list[str] = []
    handler = PoolIndexHandler(
        docroot=str(docroot),
        watch_relpath="docs",
        on_index=indexed.append,
        on_delete=deleted.append,
    )

    handler.on_deleted(MagicMock(is_directory=False, src_path=str(file_path)))

    assert indexed == []
    assert deleted == ["docs/note.txt"]


def test_handler_moved_reindexes_and_deletes_old(tmp_path: Path) -> None:
    docroot = tmp_path / "pool"
    watch_dir = docroot / "docs"
    watch_dir.mkdir(parents=True)
    old_path = watch_dir / "old.txt"
    new_path = watch_dir / "new.txt"
    new_path.write_text("moved", encoding="utf-8")

    indexed: list[str] = []
    deleted: list[str] = []
    handler = PoolIndexHandler(
        docroot=str(docroot),
        watch_relpath="docs",
        on_index=indexed.append,
        on_delete=deleted.append,
    )

    handler.on_moved(
        MagicMock(
            is_directory=False,
            src_path=str(old_path),
            dest_path=str(new_path),
        )
    )

    assert deleted == ["docs/old.txt"]
    assert indexed == ["docs/new.txt"]


def test_handler_ignores_paths_outside_watch_tree(tmp_path: Path) -> None:
    docroot = tmp_path / "pool"
    docs = docroot / "docs"
    other = docroot / "other"
    docs.mkdir(parents=True)
    other.mkdir()
    file_path = other / "secret.txt"
    file_path.write_text("hidden", encoding="utf-8")

    indexed: list[str] = []
    deleted: list[str] = []
    handler = PoolIndexHandler(
        docroot=str(docroot),
        watch_relpath="docs",
        on_index=indexed.append,
        on_delete=deleted.append,
    )

    handler.on_created(MagicMock(is_directory=False, src_path=str(file_path)))

    assert indexed == []
    assert deleted == []


def test_watcher_watch_abspath(sample_pool, test_config, tmp_path) -> None:
    docroot = tmp_path / "pool"
    subdir = docroot / "subdir"
    subdir.mkdir(parents=True)
    dbop = MagicMock(docroot=str(docroot), pool=sample_pool)
    watcher = PoolWatcher(dbop, watch_relpath="subdir")
    assert watcher.watch_abspath.endswith("/subdir")
