"""Tests for text embedding helpers and pipeline workers."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from meipi.indexing.embedding.text_embedding import (
    EmbeddingConfig,
    EmbeddingPipeline,
    TextEmbedding,
)
from meipi.indexing.model import ChunkItem, DBBgeM3Vector


@pytest.fixture
def embedder() -> TextEmbedding:
    return TextEmbedding(EmbeddingConfig(), load_model=False)


@pytest.fixture
def pipeline() -> EmbeddingPipeline:
    return EmbeddingPipeline(EmbeddingConfig(num_workers=1), pool_id=1)


def test_ingest_chunks_worker_converts_db_vector(pipeline: EmbeddingPipeline) -> None:
    chunk_queue: mp.Queue = mp.Queue()
    row = DBBgeM3Vector(doc_id=3, chunk_index=2, content="sample text")

    pipeline.ingest_chunks_worker([row], chunk_queue)

    chunk = chunk_queue.get(timeout=1)
    assert isinstance(chunk, ChunkItem)
    assert chunk.doc_id == 3
    assert chunk.chunk_index == 2
    assert chunk.content == "sample text"
    assert chunk_queue.get(timeout=1) is None


def test_ingest_chunks_worker_passes_chunk_item(pipeline: EmbeddingPipeline) -> None:
    chunk_queue: mp.Queue = mp.Queue()
    item = ChunkItem(doc_id=5, chunk_index=0, content="unchanged")

    pipeline.ingest_chunks_worker([item], chunk_queue)

    chunk = chunk_queue.get(timeout=1)
    assert chunk.doc_id == item.doc_id
    assert chunk.chunk_index == item.chunk_index
    assert chunk.content == item.content
    assert chunk_queue.get(timeout=1) is None


def test_encode_chunk_token_ids_reserves_prefix_budget(embedder: TextEmbedding) -> None:
    token_ids = embedder.encode_chunk_token_ids("word " * 2000)
    prefix_len = len(embedder._prefix_token_ids)

    assert token_ids[:prefix_len] == embedder._prefix_token_ids
    assert len(token_ids) <= embedder.config.max_length


def test_encode_chunk_token_ids_short_text(embedder: TextEmbedding) -> None:
    token_ids = embedder.encode_chunk_token_ids("hello world")
    prefix_len = len(embedder._prefix_token_ids)

    assert token_ids[:prefix_len] == embedder._prefix_token_ids
    assert len(token_ids) > prefix_len


def test_embed_token_ids_requires_loaded_model(embedder: TextEmbedding) -> None:
    with pytest.raises(RuntimeError, match="Model not loaded"):
        embedder.embed_token_ids([[1, 2, 3]])


def test_run_pipeline_raises_on_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    processes: list[FakeProcess] = []

    class FakeProcess:
        def __init__(self, target=None, args=()) -> None:
            self.name = getattr(target, "__name__", "worker")
            self.exitcode = 0
            processes.append(self)

        def start(self) -> None:
            return None

        def join(self) -> None:
            if processes:
                processes[0].exitcode = 1

    class FakeCtx:
        Queue = mp.Queue
        Process = FakeProcess

    monkeypatch.setattr(
        "meipi.indexing.embedding.text_embedding._MP_CTX",
        FakeCtx(),
    )

    pipeline = EmbeddingPipeline(EmbeddingConfig(num_workers=1), pool_id=1)

    with pytest.raises(RuntimeError, match="Embedding pipeline worker failed"):
        pipeline.run_pipeline([])
