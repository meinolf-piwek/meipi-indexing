"""Tests for single-file index workflow used by the watcher."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from meipi.indexing.cmd.main import index_file
from meipi.indexing.model import DBMeta, DBPool


@pytest.mark.asyncio
async def test_index_file_persists_inhalt_on_reindex(
    sample_pool: DBPool,
    test_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    added: list[DBMeta] = []
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False

    def capture_add(obj: DBMeta) -> None:
        added.append(obj)

    session.add.side_effect = capture_add

    dbop = MagicMock()
    dbop.pool = sample_pool
    dbop.Session = MagicMock(return_value=session)
    dbop.update_thumb_for_pic = MagicMock()
    dbop.upsert_document_chunks = MagicMock(return_value=2)

    now = datetime.now()
    new_meta = DBMeta(
        pool_id=sample_pool.id,
        path="notes.txt",
        fname="notes.txt",
        suffix=".txt",
        sort_date=now,
        fdate=now,
        fsize=12,
        clength=12,
        ctype="text/plain",
        ftype="doc",
        md_keys=["Content-Type"],
        meta_data={"Content-Type": "text/plain"},
        sha256=b"\x00" * 32,
        inhalt="updated body text",
    )

    class FakeAfop:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def file_to_db(self, rel_path: str):
            return new_meta

    monkeypatch.setattr(
        "meipi.indexing.cmd.main.DBOperations",
        lambda **kwargs: dbop,
    )
    monkeypatch.setattr(
        "meipi.indexing.cmd.main.AsyncFileOperations",
        lambda pool, config: FakeAfop(),
    )

    ok = await index_file(pool=sample_pool, rel_path="notes.txt", update_thumbs=False)

    assert ok is True
    assert len(added) == 1
    assert added[0].inhalt == "updated body text"
    session.execute.assert_called_once()
    session.flush.assert_called()
    session.commit.assert_called_once()
    dbop.upsert_document_chunks.assert_called_once_with(
        new_meta.id, "updated body text"
    )
