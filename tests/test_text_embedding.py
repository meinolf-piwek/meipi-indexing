"""Tests for text embedding helpers and pipeline workers."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from meipi.indexing.config import EmbeddingConfig
from meipi.indexing.embedding.text_embedding import TextEmbedding
from meipi.indexing.embedding_pipeline import (
    STAGE_EOF,
    DocItem,
    EmbeddingPipeline,
    _produce_ingest_chunks,
)
from meipi.indexing.model import ChunkItem


@pytest.fixture
def embedder() -> TextEmbedding:
    return TextEmbedding(EmbeddingConfig(), load_model=False)


@pytest.fixture
def pipeline() -> EmbeddingPipeline:
    return EmbeddingPipeline(EmbeddingConfig(num_workers=1), pool_id=1)


def test_ingest_chunks_worker_loads_chunks_for_doc(
    pipeline: EmbeddingPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk_queue: mp.Queue = mp.Queue()

    class FakeRow:
        doc_id = 3
        chunk_index = 2
        content = "sample text"

        def _asdict(self):
            return {
                "doc_id": self.doc_id,
                "chunk_index": self.chunk_index,
                "content": self.content,
            }

    class FakeSession:
        def execute(self, _stmt):
            class Result:
                def fetchall(self):
                    return [FakeRow()]

            return Result()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeDBOps:
        def __init__(self, pool_id: int) -> None:
            self.pool_id = pool_id

        def Session(self):
            return FakeSession()

    monkeypatch.setattr(
        "meipi.indexing.embedding_pipeline.DBOperations",
        FakeDBOps,
    )

    pipeline.ingest_chunks_worker([DocItem(id=3, inhalt="")], chunk_queue)

    chunk = chunk_queue.get(timeout=1)
    assert isinstance(chunk, ChunkItem)
    assert chunk.doc_id == 3
    assert chunk.chunk_index == 2
    assert chunk.content == "sample text"
    assert chunk_queue.get(timeout=1) == STAGE_EOF
    assert chunk_queue.empty()


def test_produce_ingest_chunks_emits_stage_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_queue: mp.Queue = mp.Queue()

    class FakeSession:
        def execute(self, _stmt):
            class Result:
                def fetchall(self):
                    return []

            return Result()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeDBOps:
        def __init__(self, pool_id: int) -> None:
            self.pool_id = pool_id

        def Session(self):
            return FakeSession()

    monkeypatch.setattr(
        "meipi.indexing.embedding_pipeline.DBOperations",
        FakeDBOps,
    )

    _produce_ingest_chunks([], pool_id=1, out_q=chunk_queue)

    assert chunk_queue.get(timeout=1) == STAGE_EOF


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
        def __init__(self, target=None, args=(), name=None) -> None:
            self.name = name or getattr(target, "__name__", "worker")
            self.exitcode: int | None = 0
            self._alive = True
            processes.append(self)

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False
            if self.exitcode is None:
                self.exitcode = -15

        def join(self, timeout: float | None = None) -> None:
            if self._alive and processes:
                processes[0].exitcode = 1
                processes[0]._alive = False
            self._alive = False

    class FakeCtx:
        Queue = mp.Queue
        Process = FakeProcess

    monkeypatch.setattr(
        "meipi.indexing.embedding_pipeline._MP_CTX",
        FakeCtx(),
    )

    pipeline = EmbeddingPipeline(EmbeddingConfig(num_workers=1), pool_id=1)

    with pytest.raises(RuntimeError, match="Embedding pipeline worker failed"):
        pipeline.run_transformers_pipeline([])


def test_run_pipeline_aborts_when_downstream_dies(monkeypatch: pytest.MonkeyPatch) -> None:
    processes: list[FakeProcess] = []

    class FakeProcess:
        def __init__(self, target=None, args=(), name=None) -> None:
            self.name = name or getattr(target, "__name__", "worker")
            self.exitcode: int | None = None
            self._alive = True
            processes.append(self)

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False
            if self.exitcode is None:
                self.exitcode = -15

        def join(self, timeout: float | None = None) -> None:
            if self.name == "produce" and self._alive:
                for process in processes:
                    if process.name == "dbwrite" and process._alive:
                        process.exitcode = 1
                        process._alive = False

    class FakeCtx:
        Queue = mp.Queue
        Process = FakeProcess

    monkeypatch.setattr(
        "meipi.indexing.embedding_pipeline._MP_CTX",
        FakeCtx(),
    )

    pipeline = EmbeddingPipeline(EmbeddingConfig(num_workers=1), pool_id=1)

    with pytest.raises(RuntimeError, match="Embedding pipeline worker failed: .*dbwrite"):
        pipeline.run_transformers_pipeline([])
