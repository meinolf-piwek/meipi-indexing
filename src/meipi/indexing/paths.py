"""Filesystem path helpers shared by watcher and database operations."""

from __future__ import annotations

import os


def normalize_rel_path(path: str) -> str:
    """Normalize a relative path for stable DB lookups."""
    return os.path.normpath(path).replace("\\", "/")


def resolve_watch_relpath(docroot: str, watch_path: str) -> str:
    """Return *watch_path* relative to *docroot* (accepts absolute paths under docroot)."""
    docroot_abs = os.path.abspath(docroot)
    if os.path.isabs(watch_path):
        watch_abs = os.path.abspath(watch_path)
        if watch_abs == docroot_abs:
            return "."
        try:
            rel = os.path.relpath(watch_abs, docroot_abs)
        except ValueError as exc:
            raise ValueError(
                f"Watch path {watch_path!r} is not under docroot {docroot_abs!r}. "
                "Set IND_DOCROOT or pass --docroot to the indexed filesystem root."
            ) from exc
        if rel == ".." or rel.startswith(".." + os.sep):
            raise ValueError(
                f"Watch path {watch_path!r} is not under docroot {docroot_abs!r}. "
                "Set IND_DOCROOT or pass --docroot to the indexed filesystem root."
            )
        return normalize_rel_path(rel)
    return normalize_rel_path(watch_path)


def is_under_watch_tree(rel_path: str, watch_relpath: str) -> bool:
    """Return whether ``rel_path`` lies inside the watched subtree."""
    rel_path = normalize_rel_path(rel_path)
    watch_relpath = normalize_rel_path(watch_relpath)
    if watch_relpath in ("", "."):
        return True
    return rel_path == watch_relpath or rel_path.startswith(watch_relpath + "/")


def list_watch_files(docroot: str, watch_relpath: str) -> set[str]:
    """Return relative file paths under ``watch_relpath`` within ``docroot``."""
    watch_abspath = os.path.join(docroot, watch_relpath)
    if not os.path.isdir(watch_abspath):
        raise ValueError(f"Watch path is not a directory: {watch_abspath}")

    paths: set[str] = set()
    for dirpath, _, filenames in os.walk(watch_abspath):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            rel_path = normalize_rel_path(os.path.relpath(filepath, docroot))
            if is_under_watch_tree(rel_path, watch_relpath):
                paths.add(rel_path)
    return paths
