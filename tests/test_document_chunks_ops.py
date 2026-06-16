"""Tests for storing document chunks."""


def test_replace_document_chunks(mock_db_operations) -> None:
    ops, session = mock_db_operations
    contents = ["First chunk", "Second chunk"]

    count = ops.replace_document_chunks(doc_id=4, contents=contents)

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

    count = ops.replace_document_chunks(doc_id=4, contents=[])

    assert count == 0
    session.execute.assert_called_once()
    session.add_all.assert_not_called()
    session.commit.assert_called_once()


def test_insert_document_chunks(mock_db_operations) -> None:
    ops, session = mock_db_operations

    count = ops.insert_document_chunks(doc_id=4, contents=["First chunk"])

    assert count == 1
    session.execute.assert_not_called()
    session.add_all.assert_called_once()
    added = session.add_all.call_args.args[0]
    assert added[0].doc_id == 4
    assert added[0].chunk_index == 0
    session.commit.assert_called_once()
