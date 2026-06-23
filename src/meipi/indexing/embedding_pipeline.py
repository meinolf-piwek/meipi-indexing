"""Multiprocessing pipeline for document chunking, embedding, and DB persistence.

Each pipeline stage runs in its own coordinator process. Parallel stages use an
internal worker pool; the coordinator reads from the upstream queue until
``STAGE_EOF``, fans work out to the pool, and emits a single ``STAGE_EOF`` to
the downstream queue when finished.

The main process only starts stage coordinators and waits for them, polling for
non-zero exit codes or premature loss of a downstream stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Sequence
import multiprocessing as mp
from multiprocessing.process import BaseProcess

from tqdm.auto import tqdm
import sqlalchemy as sa

from .config import EmbeddingConfig
from .embedding.flag_embedding import FlagEmbedding
from .embedding.text_chunking import DocumentChunker
from .embedding.text_embedding import TextEmbedding
from .model import ChunkItem, DBBgeM3Vector
from .operations import DBOperations

_MP_CTX = mp.get_context("spawn")
_POLL_INTERVAL = 0.5
STAGE_EOF = "__PIPELINE_STAGE_EOF__"

_token_embedder: TextEmbedding | None = None
_embed_model: TextEmbedding | None = None
_flag_embedder: FlagEmbedding | None = None


def _queue_items(in_q: mp.Queue) -> Iterator[ChunkItem]:
    while True:
        item = in_q.get()
        if item == STAGE_EOF:
            return
        yield item


def _terminate_processes(processes: Sequence[BaseProcess]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.is_alive():
            process.join(timeout=5)


def _raise_pipeline_failure(
    processes: Sequence[BaseProcess],
    *,
    reason: str,
) -> None:
    _terminate_processes(processes)
    failed = [p for p in processes if p.exitcode not in (None, 0)]
    if failed:
        details = ", ".join(
            f"{p.name or 'stage'}(exitcode={p.exitcode})" for p in failed
        )
        raise RuntimeError(f"Embedding pipeline worker failed: {details}")
    raise RuntimeError(f"Embedding pipeline aborted: {reason}")


def _check_processes(processes: Sequence[BaseProcess]) -> None:
    failed = [p for p in processes if p.exitcode not in (None, 0)]
    if failed:
        details = ", ".join(
            f"{p.name or 'stage'}(exitcode={p.exitcode})" for p in failed
        )
        _raise_pipeline_failure(
            processes,
            reason=f"worker failed: {details}",
        )


def _require_alive(
    workers: Sequence[BaseProcess],
    processes: Sequence[BaseProcess],
    *,
    stage: str,
) -> None:
    if any(worker.is_alive() for worker in workers):
        return
    _raise_pipeline_failure(
        processes,
        reason=f"all {stage} workers exited before pipeline shutdown",
    )


def _wait_stages(stages: Sequence[BaseProcess]) -> None:
    stage_list = list(stages)
    while any(stage.is_alive() for stage in stage_list):
        _check_processes(stage_list)
        for index, stage in enumerate(stage_list[:-1]):
            if not stage.is_alive():
                continue
            downstream = stage_list[index + 1 :]
            label = downstream[0].name or "downstream"
            _require_alive(downstream, stage_list, stage=label)
        for stage in stage_list:
            if stage.is_alive():
                stage.join(timeout=_POLL_INTERVAL)
    _check_processes(stage_list)


def _init_token_worker(config: EmbeddingConfig) -> None:
    global _token_embedder
    _token_embedder = TextEmbedding(config, load_model=False)


def _tokenize_chunk(chunk: ChunkItem) -> tuple[ChunkItem, list[int]]:
    if _token_embedder is None:
        raise RuntimeError("tokenize worker not initialized")
    return chunk, _token_embedder.encode_chunk_token_ids(chunk.content)


def _tokenize_stage(
    config: EmbeddingConfig,
    in_q: mp.Queue,
    out_q: mp.Queue,
    num_workers: int,
) -> None:
    with _MP_CTX.Pool(
        num_workers,
        initializer=_init_token_worker,
        initargs=(config,),
    ) as pool:
        with tqdm(desc="Tokenizing chunks", leave=True) as progress:
            for result in pool.imap(_tokenize_chunk, _queue_items(in_q), chunksize=1):
                out_q.put(result)
                progress.update(1)
    out_q.put(STAGE_EOF)


def _init_embed_worker(config: EmbeddingConfig) -> None:
    global _embed_model
    _embed_model = TextEmbedding(config, load_model=True)


def _embed_transformers_batch(
    batch: list[tuple[ChunkItem, list[int]]],
) -> list[tuple[ChunkItem, object]]:
    if _embed_model is None:
        raise RuntimeError("embed worker not initialized")
    token_id_batch = [token_ids for _, token_ids in batch]
    embeddings = _embed_model.embed_token_ids(token_id_batch)
    return [(chunk, vector) for (chunk, _), vector in zip(batch, embeddings)]


def _embed_transformers_stage(
    config: EmbeddingConfig,
    in_q: mp.Queue,
    out_q: mp.Queue,
) -> None:
    with _MP_CTX.Pool(1, initializer=_init_embed_worker, initargs=(config,)) as pool:
        batch: list[tuple[ChunkItem, list[int]]] = []
        with tqdm(desc="Embedding chunks", leave=True) as progress:
            while True:
                item = in_q.get()
                if item == STAGE_EOF:
                    break
                batch.append(item)
                if len(batch) < config.batch_size:
                    continue
                for entry in pool.apply(_embed_transformers_batch, (batch,)):
                    out_q.put(entry)
                    progress.update(1)
                batch = []
            if batch:
                for entry in pool.apply(_embed_transformers_batch, (batch,)):
                    out_q.put(entry)
                    progress.update(1)
    out_q.put(STAGE_EOF)


def _init_flag_worker(config: EmbeddingConfig) -> None:
    global _flag_embedder
    _flag_embedder = FlagEmbedding(config)


def _flag_embed_chunk(chunk: ChunkItem) -> tuple[ChunkItem, object]:
    if _flag_embedder is None:
        raise RuntimeError("flag embed worker not initialized")
    return chunk, _flag_embedder.embed(chunk.content)


def _embed_flag_stage(
    config: EmbeddingConfig,
    in_q: mp.Queue,
    out_q: mp.Queue,
) -> None:
    with _MP_CTX.Pool(1, initializer=_init_flag_worker, initargs=(config,)) as pool:
        with tqdm(desc="Embedding chunks", leave=True) as progress:
            for result in pool.imap(_flag_embed_chunk, _queue_items(in_q), chunksize=1):
                out_q.put(result)
                progress.update(1)
    out_q.put(STAGE_EOF)


def _dbwrite_stage(pool_id: int, in_q: mp.Queue) -> None:
    dbop = DBOperations(pool_id=pool_id)
    with dbop.Session() as session:
        with tqdm(desc="Writing chunks to database", leave=True) as progress:
            while True:
                item = in_q.get()
                if item == STAGE_EOF:
                    break
                chunk, vector = item
                dbrow = session.get(DBBgeM3Vector, (chunk.doc_id, chunk.chunk_index))
                if dbrow is None:
                    dbrow = DBBgeM3Vector(
                        doc_id=chunk.doc_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                    )
                    session.add(dbrow)
                dbrow.vector = vector
                progress.update(1)
        session.flush()
        session.commit()


def _dummy_dbwrite_stage(in_q: mp.Queue) -> None:
    res = []
    with tqdm(desc="Writing chunks to database", leave=True) as progress:
        while True:
            item = in_q.get()
            if item == STAGE_EOF:
                break
            chunk, vector = item
            res.append((chunk.doc_id, chunk.chunk_index, vector))
            progress.update(1)
    print(f"Dummy writing {len(res)} chunks")
    if res:
        print(res[0])
        print(res[-1])


def _produce_ingest_chunks(
    docs: Sequence[DocItem],
    pool_id: int,
    out_q: mp.Queue,
) -> None:
    with DBOperations(pool_id=pool_id).Session() as session:
        for doc in tqdm(docs, desc="Ingesting chunks", leave=True):
            chunk_rows = session.execute(
                sa.select(
                    DBBgeM3Vector.doc_id,
                    DBBgeM3Vector.chunk_index,
                    DBBgeM3Vector.content,
                ).where(DBBgeM3Vector.doc_id == doc.id)
            ).fetchall()
            for chunk_row in chunk_rows:
                out_q.put(ChunkItem(**chunk_row._asdict()))
    out_q.put(STAGE_EOF)


def _produce_chunk_documents(
    docs: Sequence[DocItem],
    config: EmbeddingConfig,
    out_q: mp.Queue,
) -> None:
    chunker = DocumentChunker(config)
    for doc in tqdm(docs, desc="Chunking documents", leave=True):
        for chunk in chunker.chunk_doc(doc.id, doc.inhalt):
            out_q.put(chunk)
    out_q.put(STAGE_EOF)


@dataclass
class DocItem:
    id: int
    inhalt: str


class EmbeddingPipeline:
    def __init__(self, config: EmbeddingConfig, pool_id: int):
        self.config = config
        self.pool_id = pool_id
        self.mp_ctx = mp.get_context("spawn")

    def ingest_chunks_worker(
        self,
        docs: list[DocItem],
        chunk_queue: mp.Queue,
    ) -> None:
        """Load existing chunks from the database and signal stage completion."""
        _produce_ingest_chunks(docs, self.pool_id, chunk_queue)

    def chunking_worker(
        self,
        docs: list[DocItem],
        chunk_queue: mp.Queue,
    ) -> None:
        """Chunk documents in-process and signal stage completion."""
        _produce_chunk_documents(docs, self.config, chunk_queue)

    def run_transformers_pipeline(self, docs: Sequence[DocItem]) -> None:
        start_time = datetime.now()
        chunk_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        token_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        embedding_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        num_token_workers = self.config.num_workers

        print("Starting ingest process at", start_time)
        token_stage = _MP_CTX.Process(
            target=_tokenize_stage,
            args=(self.config, chunk_queue, token_queue, num_token_workers),
            name="tokenize",
        )
        embed_stage = _MP_CTX.Process(
            target=_embed_transformers_stage,
            args=(self.config, token_queue, embedding_queue),
            name="embed",
        )
        dbwrite_stage = _MP_CTX.Process(
            target=_dbwrite_stage,
            args=(self.pool_id, embedding_queue),
            name="dbwrite",
        )
        produce_stage = _MP_CTX.Process(
            target=_produce_ingest_chunks,
            args=(docs, self.pool_id, chunk_queue),
            name="produce",
        )

        for stage in (token_stage, embed_stage, dbwrite_stage):
            stage.start()
        produce_stage.start()

        _wait_stages([produce_stage, token_stage, embed_stage, dbwrite_stage])

        end_time = datetime.now()
        print(
            "End time:",
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Start time:",
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        print("Pipeline finished in", (end_time - start_time), "seconds")

    def run_flag_pipeline(
        self,
        docs: list[DocItem],
        create_chunks: bool = True,
        test: bool = True,
    ) -> None:
        start_time = datetime.now()
        chunk_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        embedding_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)

        if create_chunks:
            produce_target = _produce_chunk_documents
            produce_args: tuple = (docs, self.config, chunk_queue)
        else:
            produce_target = _produce_ingest_chunks
            produce_args = (docs, self.pool_id, chunk_queue)

        embed_stage = _MP_CTX.Process(
            target=_embed_flag_stage,
            args=(self.config, chunk_queue, embedding_queue),
            name="embed",
        )
        dbwrite_target = _dummy_dbwrite_stage if test else _dbwrite_stage
        dbwrite_args: tuple = (
            (embedding_queue,) if test else (self.pool_id, embedding_queue)
        )
        dbwrite_stage = _MP_CTX.Process(
            target=dbwrite_target,
            args=dbwrite_args,
            name="dbwrite",
        )
        produce_stage = _MP_CTX.Process(
            target=produce_target,
            args=produce_args,
            name="produce",
        )

        for stage in (embed_stage, dbwrite_stage):
            stage.start()
        produce_stage.start()

        _wait_stages([produce_stage, embed_stage, dbwrite_stage])

        end_time = datetime.now()
        print(
            "End time:",
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Start time:",
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        print("Pipeline finished in", (end_time - start_time), "seconds")
