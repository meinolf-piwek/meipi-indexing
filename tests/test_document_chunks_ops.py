"""Tests for storing document chunks."""

from unittest.mock import MagicMock

from meipi.indexing.document_chunks import DocumentChunk
from meipi.indexing.model import DBBgeM3Vector


def test_replace_document_chunks(mock_db_operations) -> None:
    ops, session = mock_db_operations
    chunks = [
        DocumentChunk(chunk_index=0, content="First chunk", meta_id=10),
        DocumentChunk(chunk_index=1, content="Second chunk", meta_id=10),
    ]

    count = ops.replace_document_chunks(doc_id=4, chunks=chunks)

    assert count == 2
    session.execute.assert_called_once()
    session.add_all.assert_called_once()
    added = session.add_all.call_args.args[0]
    assert len(added) == 2
    assert added[0].doc_id == 4
    assert added[0].chunk_index == 0
    assert added[1].content == "Second chunk"
    session.commit.assert_called_once()


def test_replace_document_chunks_empty_list(mock_db_operations) -> None:
    ops, session = mock_db_operations

    count = ops.replace_document_chunks(doc_id=4, chunks=[])

    assert count == 0
    session.execute.assert_called_once()
    session.add_all.assert_not_called()
    session.commit.assert_called_once()
