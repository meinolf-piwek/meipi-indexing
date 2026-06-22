"""Filesystem watcher that keeps the index database in sync with a directory tree."""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .cmd.main import index_file
from .config import Config, resolve_config
from .operations import DBOperations, WATCHER_TABLES
from .paths import (
    ensure_watch_directory,
    is_under_watch_tree,
    list_watch_files,
    normalize_rel_path,
    resolve_watch_relpath,
    watch_abspath,
)

__all__ = [
    "PoolWatcher",
    "PoolIndexHandler",
    "SyncReport",
    "check_pool_sync",
    "format_sync_report",
    "is_under_watch_tree",
    "list_watch_files",
    "normalize_rel_path",
]


class PoolIndexHandler(FileSystemEventHandler):
    """Handle filesystem events for one datapool subtree."""

    def __init__(
        self,
        *,
        docroot: str,
        watch_relpath: str,
        on_index: Callable[[str], None],
        on_delete: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.docroot = os.path.abspath(docroot)
        self.watch_relpath = watch_relpath
        self.on_index = on_index
        self.on_delete = on_delete

    def _rel_path(self, abspath: str) -> str | None:
        try:
            rel_path = normalize_rel_path(os.path.relpath(abspath, self.docroot))
        except ValueError:
            return None
        if rel_path.startswith(".."):
            return None
        if not is_under_watch_tree(rel_path, self.watch_relpath):
            return None
        return rel_path

    def _handle_path(self, abspath: str, *, index: bool) -> None:
        if not os.path.isfile(abspath):
            return
        rel_path = self._rel_path(abspath)
        if rel_path is None:
            return
        if index:
            self.on_index(rel_path)
        else:
            self.on_delete(rel_path)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle_path(event.src_path, index=True)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle_path(event.src_path, index=True)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        rel_path = self._rel_path(event.src_path)
        if rel_path is not None:
            self.on_delete(rel_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_rel = self._rel_path(event.src_path)
        if src_rel is not None:
            self.on_delete(src_rel)
        self._handle_path(event.dest_path, index=True)


@dataclass(frozen=True, slots=True)
class SyncReport:
    """Filesystem/database sync check (report only; no changes applied)."""

    schema: str
    pool_id: int
    docroot: str
    watch_relpath: str
    schema_tables: dict[str, int]
    tables_missing: tuple[str, ...]
    filemeta_rows: int
    pool_indexed_count: int
    fs_count: int
    watch_indexed_count: int
    in_sync: tuple[str, ...]
    only_on_disk: tuple[str, ...]
    only_in_db: tuple[str, ...]

    @property
    def has_differences(self) -> bool:
        return bool(self.tables_missing or self.only_in_db or self.only_on_disk)

    @property
    def is_in_sync(self) -> bool:
        """True when the watch tree matches the database and required tables exist."""
        return not self.has_differences and not self.watch_path_mismatch

    @property
    def watch_path_mismatch(self) -> bool:
        """True when the watch path overlaps neither disk files nor indexed paths."""
        return (
            self.pool_indexed_count > 0
            and self.watch_indexed_count == 0
            and self.fs_count == 0
        )

    def as_dict(self, *, verbose: bool = False) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "schema": self.schema,
            "pool_id": self.pool_id,
            "docroot": self.docroot,
            "watch_relpath": self.watch_relpath,
            "watch_path_mismatch": self.watch_path_mismatch,
            "schema_table_count": len(self.schema_tables),
            "filemeta_rows": self.filemeta_rows,
            "pool_indexed_count": self.pool_indexed_count,
            "tables_missing_count": len(self.tables_missing),
            "has_differences": self.has_differences,
            "filesystem": {
                "watch_tree_file_count": self.fs_count,
                "only_on_disk_count": len(self.only_on_disk),
            },
            "database": {
                "watch_tree_indexed_count": self.watch_indexed_count,
                "only_in_db_count": len(self.only_in_db),
            },
            "matched_count": len(self.in_sync),
        }
        if not verbose:
            return summary
        return {
            **summary,
            "tables": self.schema_tables,
            "tables_missing": list(self.tables_missing),
            "filesystem": {
                "watch_tree_file_count": self.fs_count,
                "only_on_disk": list(self.only_on_disk),
            },
            "database": {
                "watch_tree_indexed_count": self.watch_indexed_count,
                "only_in_db": list(self.only_in_db),
            },
            "matched": {
                "count": len(self.in_sync),
                "paths": list(self.in_sync),
            },
        }

    def format(self, *, verbose: bool = False) -> str:
        return format_sync_report(self, verbose=verbose)


def format_sync_report(report: SyncReport, *, verbose: bool = False) -> str:
    """Human-readable sync check report (summary by default)."""
    lines = [
        f"Schema: {report.schema} ({len(report.schema_tables)} tables)",
        (
            f"filemeta: {report.filemeta_rows:,} rows in schema · "
            f"pool {report.pool_id}: {report.pool_indexed_count:,} indexed paths"
        ),
    ]
    if report.tables_missing:
        lines.append(f"Missing tables: {len(report.tables_missing)}")

    lines.append(
        f"Watch tree: {report.fs_count:,} files on disk · "
        f"{report.watch_indexed_count:,} indexed in tree · "
        f"{len(report.in_sync):,} matched"
    )
    if report.only_in_db:
        lines.append(f"Only in DB: {len(report.only_in_db)}")
    if report.only_on_disk:
        lines.append(f"Only on disk: {len(report.only_on_disk)}")

    if verbose:
        if report.schema_tables:
            lines.append("Table rows:")
            for name, count in sorted(report.schema_tables.items()):
                lines.append(f"  {name:<20}: {count:>10,} rows")
        if report.tables_missing:
            lines.append(f"Missing: {', '.join(report.tables_missing)}")
        if report.only_in_db:
            lines.append("Only in DB:")
            lines.extend(f"  - {path}" for path in report.only_in_db)
        if report.only_on_disk:
            lines.append("Only on disk:")
            lines.extend(f"  - {path}" for path in report.only_on_disk)
        if report.in_sync:
            lines.append("Matched:")
            lines.extend(f"  - {path}" for path in report.in_sync)

    if report.watch_path_mismatch:
        lines.append(
            f"Watch path {report.watch_relpath!r} under docroot {report.docroot!r} "
            "matches no files on disk and no indexed paths for this pool"
        )
        lines.append(
            "Hint: RELPATH is relative to the pool docroot from serverpools; "
            "set IND_SERVER_NAME or pass --server_name"
        )
    elif report.has_differences:
        lines.append("Differences found (no changes applied)")
    else:
        lines.append("Filesystem and database are in sync")
    return "\n".join(lines)


def check_pool_sync(
    dbop: DBOperations,
    *,
    watch_relpath: str = ".",
) -> SyncReport:
    """Compare the watched filesystem tree with the database (report only)."""
    watch_relpath = ensure_watch_directory(dbop.docroot, watch_relpath)
    schema_info = dbop.schema_info()
    existing_tables = set(schema_info["tables"])
    tables_missing = tuple(
        entity.__tablename__
        for entity in WATCHER_TABLES
        if entity.__tablename__ not in existing_tables
    )

    filemeta_rows = int(schema_info["tables"].get("filemeta", 0))
    pool_indexed_count = dbop.count_pool_filemeta()

    fs_paths = list_watch_files(dbop.docroot, watch_relpath)
    db_paths = dbop.list_pool_file_paths(watch_relpath)
    in_sync = sorted(fs_paths & db_paths)
    only_in_db = sorted(db_paths - fs_paths)
    only_on_disk = sorted(fs_paths - db_paths)

    return SyncReport(
        schema=schema_info["schema"],
        pool_id=dbop.pool.id,
        docroot=dbop.docroot,
        watch_relpath=watch_relpath,
        schema_tables=dict(schema_info["tables"]),
        tables_missing=tables_missing,
        filemeta_rows=filemeta_rows,
        pool_indexed_count=pool_indexed_count,
        fs_count=len(fs_paths),
        watch_indexed_count=len(db_paths),
        in_sync=tuple(in_sync),
        only_on_disk=tuple(only_on_disk),
        only_in_db=tuple(only_in_db),
    )


class PoolWatcher:
    """Watch a datapool directory and sync file changes to PostgreSQL."""

    def __init__(
        self,
        dbop: DBOperations,
        *,
        watch_relpath: str = ".",
        debounce_seconds: float = 1.0,
        update_thumbs: bool = True,
        config: Config | None = None,
    ) -> None:
        config = resolve_config(config)
        self.dbop = dbop
        self.watch_relpath = ensure_watch_directory(dbop.docroot, watch_relpath)
        self.debounce_seconds = debounce_seconds
        self.update_thumbs = update_thumbs
        self.config = config
        self.logger = config.logger
        self._pending_index: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._observer: Observer | None = None

    @property
    def watch_abspath(self) -> str:
        return watch_abspath(self.dbop.docroot, self.watch_relpath)

    def _schedule_index(self, rel_path: str) -> None:
        with self._lock:
            timer = self._pending_index.pop(rel_path, None)
            if timer is not None:
                timer.cancel()

            def _run() -> None:
                with self._lock:
                    self._pending_index.pop(rel_path, None)
                self._index_file(rel_path)

            timer = threading.Timer(self.debounce_seconds, _run)
            self._pending_index[rel_path] = timer
            timer.start()

    def _index_file(self, rel_path: str) -> None:
        self.logger.info("Indexing %s", rel_path)
        try:
            indexed = asyncio.run(
                index_file(
                    pool=self.dbop.pool,
                    rel_path=rel_path,
                    update_thumbs=self.update_thumbs,
                    config=self.config,
                )
            )
        except Exception:
            self.logger.exception("Failed to index %s", rel_path)
            return
        if not indexed:
            self.logger.warning("No index entry created for %s", rel_path)

    def _delete_file(self, rel_path: str) -> None:
        with self._lock:
            timer = self._pending_index.pop(rel_path, None)
            if timer is not None:
                timer.cancel()
        self.logger.info("Removing index entry for %s", rel_path)
        try:
            self.dbop.delete_file_meta(rel_path)
        except Exception:
            self.logger.exception("Failed to delete index entry for %s", rel_path)

    def check_at_startup(self) -> SyncReport:
        """Check database tables and index rows against the watched tree."""
        return check_pool_sync(self.dbop, watch_relpath=self.watch_relpath)

    def run(self, *, block: bool = True, startup_check: bool = False) -> SyncReport | None:
        """Start watching until interrupted or ``block=False``."""
        report: SyncReport | None = None
        if startup_check:
            report = self.check_at_startup()
            if not report.is_in_sync:
                self.logger.error(
                    "Pool %s out of sync at %r; watcher not started",
                    self.dbop.pool.id,
                    self.watch_relpath,
                )
                return report
        watch_path = Path(self.watch_abspath)
        if not watch_path.is_dir():
            raise ValueError(f"Watch path is not a directory: {watch_path}")

        handler = PoolIndexHandler(
            docroot=self.dbop.docroot,
            watch_relpath=self.watch_relpath,
            on_index=self._schedule_index,
            on_delete=self._delete_file,
        )
        observer = Observer()
        observer.schedule(handler, str(watch_path), recursive=True)
        self._observer = observer
        observer.start()
        self.logger.info(
            "Watching pool %s at %s (debounce=%ss)",
            self.dbop.pool.id,
            watch_path,
            self.debounce_seconds,
        )
        if block:
            try:
                observer.join()
            except KeyboardInterrupt:
                self.stop()
        return report

    def stop(self) -> None:
        """Stop the observer and cancel pending index timers."""
        with self._lock:
            for timer in self._pending_index.values():
                timer.cancel()
            self._pending_index.clear()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
