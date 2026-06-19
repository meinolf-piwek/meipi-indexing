"""Tests for storing document chunks."""

from unittest.mock import MagicMock

from meipi.indexing.model import ChunkItem


def test_upsert_document_chunks(mock_db_operations) -> None:
    ops, session = mock_db_operations
    chunker = MagicMock()
    chunker.chunk_doc.return_value = [
        ChunkItem(doc_id=4, chunk_index=0, content="First chunk"),
        ChunkItem(doc_id=4, chunk_index=1, content="Second chunk"),
    ]
    ops._document_chunker = MagicMock(return_value=chunker)

    count = ops.upsert_document_chunks(4, "document text")

    assert count == 2
    chunker.chunk_doc.assert_called_once_with(4, "document text")
    session.execute.assert_called_once()
    session.add_all.assert_called_once()
    added = session.add_all.call_args.args[0]
    assert len(added) == 2
    assert added[0].doc_id == 4
    assert added[0].chunk_index == 0
    assert added[1].content == "Second chunk"
    session.commit.assert_called_once()


def test_upsert_document_chunks_empty_text(mock_db_operations) -> None:
    ops, session = mock_db_operations
    chunker = MagicMock()
    chunker.chunk_doc.return_value = []
    ops._document_chunker = MagicMock(return_value=chunker)

    count = ops.upsert_document_chunks(4, "")

    assert count == 0
    session.execute.assert_called_once()
    session.add_all.assert_called_once_with([])
    session.commit.assert_called_once()
