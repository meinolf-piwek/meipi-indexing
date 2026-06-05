"""Filesystem watcher that keeps the index database in sync with a directory tree."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .cmd.main import index_file
from .config import Config, resolve_config
from .operations import DBOperations


def normalize_rel_path(path: str) -> str:
    """Normalize a relative path for stable DB lookups."""
    return os.path.normpath(path).replace("\\", "/")


def is_under_watch_tree(rel_path: str, watch_relpath: str) -> bool:
    """Return whether ``rel_path`` lies inside the watched subtree."""
    rel_path = normalize_rel_path(rel_path)
    watch_relpath = normalize_rel_path(watch_relpath)
    if watch_relpath in ("", "."):
        return True
    return rel_path == watch_relpath or rel_path.startswith(watch_relpath + os.sep)


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
        self.watch_relpath = watch_relpath
        self.debounce_seconds = debounce_seconds
        self.update_thumbs = update_thumbs
        self.config = config
        self.logger = config.logger
        self._pending_index: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._observer: Observer | None = None

    @property
    def watch_abspath(self) -> str:
        return os.path.join(self.dbop.docroot, self.watch_relpath)

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

    def run(self, *, block: bool = True) -> None:
        """Start watching until interrupted or ``block=False``."""
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
