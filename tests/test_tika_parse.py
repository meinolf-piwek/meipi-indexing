"""Tests for AsyncFileOperations.tika_parse and file_to_db."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from meipi.indexing.config import Config
from meipi.indexing.model import DBPool
from meipi.indexing.operations import AsyncFileOperations
from tika_client import TikaKey


def _tika_response(data: dict) -> MagicMock:
    response = MagicMock()
    response.data = {TikaKey.Content: "body text", **data}
    return response


@pytest.fixture
def async_file_ops(test_config: Config, sample_pool: DBPool) -> AsyncFileOperations:
    return AsyncFileOperations(sample_pool, config=test_config, skip_ocr=True)


@pytest.mark.asyncio
async def test_tika_parse_without_dcterms_created(async_file_ops, sample_pool, monkeypatch):
    rel_path = "notes.txt"
    (Path(async_file_ops.docroot) / rel_path).write_text("hello", encoding="utf-8")

    monkeypatch.setattr(
        async_file_ops.rmeta.as_text,
        "from_file",
        AsyncMock(return_value=[_tika_response({"Content-Type": "text/plain", "Content-Length": 5})]),
    )

    dbmeta, content = await async_file_ops.tika_parse(rel_path)

    assert dbmeta is not None
    assert content == "body text"
    assert dbmeta.ftype == "doc"
    assert dbmeta.path == rel_path
    assert dbmeta.sort_date == dbmeta.fdate


@pytest.mark.asyncio
async def test_tika_parse_with_string_dcterms_created(async_file_ops, sample_pool, monkeypatch):
    rel_path = "dated.txt"
    (Path(async_file_ops.docroot) / rel_path).write_text("dated", encoding="utf-8")
    created = "2020-03-15T10:30:00"

    monkeypatch.setattr(
        async_file_ops.rmeta.as_text,
        "from_file",
        AsyncMock(
            return_value=[
                _tika_response(
                    {
                        "Content-Type": "text/plain",
                        "Content-Length": 5,
                        "dcterms:created": created,
                    }
                )
            ]
        ),
    )

    dbmeta, _ = await async_file_ops.tika_parse(rel_path)

    assert dbmeta is not None
    assert dbmeta.sort_date == datetime.fromisoformat(created)


@pytest.mark.asyncio
async def test_tika_parse_with_list_dcterms_created(async_file_ops, sample_pool, monkeypatch):
    rel_path = "list-date.txt"
    (Path(async_file_ops.docroot) / rel_path).write_text("x", encoding="utf-8")
    created = "2019-01-02T08:00:00"

    monkeypatch.setattr(
        async_file_ops.rmeta.as_text,
        "from_file",
        AsyncMock(
            return_value=[
                _tika_response(
                    {
                        "Content-Type": "text/plain",
                        "Content-Length": 1,
                        "dcterms:created": [created],
                    }
                )
            ]
        ),
    )

    dbmeta, _ = await async_file_ops.tika_parse(rel_path)

    assert dbmeta is not None
    assert dbmeta.sort_date == datetime.fromisoformat(created)


@pytest.mark.asyncio
async def test_tika_parse_bad_dcterms_falls_back_to_fdate(async_file_ops, sample_pool, monkeypatch):
    rel_path = "bad-date.txt"
    (Path(async_file_ops.docroot) / rel_path).write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        async_file_ops.rmeta.as_text,
        "from_file",
        AsyncMock(
            return_value=[
                _tika_response(
                    {
                        "Content-Type": "text/plain",
                        "Content-Length": 1,
                        "dcterms:created": "not-an-iso-date",
                    }
                )
            ]
        ),
    )

    dbmeta, _ = await async_file_ops.tika_parse(rel_path)

    assert dbmeta is not None
    assert dbmeta.sort_date == dbmeta.fdate


@pytest.mark.asyncio
async def test_tika_parse_strips_leading_trailing_newlines(async_file_ops, monkeypatch):
    rel_path = "padded.txt"
    (Path(async_file_ops.docroot) / rel_path).write_text("line one\nline two", encoding="utf-8")

    monkeypatch.setattr(
        async_file_ops.rmeta.as_text,
        "from_file",
        AsyncMock(
            return_value=[
                _tika_response(
                    {
                        "Content-Type": "text/plain",
                        "Content-Length": 17,
                        TikaKey.Content: "\n\n\nline one\nline two\n\n",
                    }
                )
            ]
        ),
    )

    dbmeta, content = await async_file_ops.tika_parse(rel_path)

    assert dbmeta is not None
    assert content == "line one line two"
    assert dbmeta.inhalt == "line one line two"


@pytest.mark.asyncio
async def test_tika_parse_cleans_css_and_js_from_inhalt(async_file_ops, monkeypatch):
    rel_path = "page.html"
    (Path(async_file_ops.docroot) / rel_path).write_text("<html></html>", encoding="utf-8")
    raw = (
        ".navi { font-size: 11px; font-family: Arial; }\n\n"
        "Vertrag über Dienstleistungen.\n\n"
        "try { window.FB.init(); } catch (e) {}"
    )

    monkeypatch.setattr(
        async_file_ops.rmeta.as_text,
        "from_file",
        AsyncMock(
            return_value=[
                _tika_response(
                    {
                        "Content-Type": "text/html",
                        "Content-Length": len(raw),
                        TikaKey.Content: raw,
                    }
                )
            ]
        ),
    )

    dbmeta, content = await async_file_ops.tika_parse(rel_path)

    assert dbmeta is not None
    assert "font-size" not in content
    assert "window.FB" not in content
    assert "Vertrag über Dienstleistungen." in content
    assert dbmeta.inhalt == content


@pytest.mark.asyncio
async def test_file_to_db_attaches_doc(async_file_ops, sample_pool, monkeypatch):
    rel_path = "a.txt"
    (Path(async_file_ops.docroot) / rel_path).write_text("content", encoding="utf-8")

    monkeypatch.setattr(
        async_file_ops.rmeta.as_text,
        "from_file",
        AsyncMock(return_value=[_tika_response({"Content-Type": "text/plain", "Content-Length": 7})]),
    )

    dbmeta = await async_file_ops.file_to_db(rel_path)

    assert dbmeta is not None
    assert dbmeta.doc is not None
    assert dbmeta.pic is None
    assert dbmeta.vid is None
    assert dbmeta.inhalt == "body text"
