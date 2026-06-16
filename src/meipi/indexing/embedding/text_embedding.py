from __future__ import annotations

import queue
from dataclasses import dataclass
from queue import Empty
from typing import Iterator, Sequence
import threading

from tqdm.auto import tqdm
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ..model import DBBgeM3Vector

_MP_CTX = mp.get_context("spawn")


@dataclass(frozen=True, slots=True)
class ChunkItem:
    doc_id: int
    chunk_index: int
    content: str


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    max_length: int = 512
    normalize: bool = True
    use_fp16: bool = True
    num_workers: int = 8
    max_queue_size: int = 2000


def _token_worker(
    chunk_queue: mp.Queue,
    token_queue: mp.Queue,
    config: EmbeddingConfig,
) -> None:
    TextEmbedding(config, load_model=False).token_worker_loop(chunk_queue, token_queue)


def _embedding_worker(
    token_queue: mp.Queue,
    embedding_queue: mp.Queue,
    config: EmbeddingConfig,
    num_token_workers: int,
) -> None:
    TextEmbedding(config).embedding_worker_loop(
        token_queue, embedding_queue, num_token_workers
    )


class TextEmbedding:
    PREFIX = "passage: "

    def __init__(self, config: EmbeddingConfig, *, load_model: bool = True):
        self.config = config
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.model_name
        )
        self.model: PreTrainedModel | None = None
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        model = AutoModel.from_pretrained(self.config.model_name)
        model.to(self.config.device)  # type: ignore[arg-type]
        if self.config.use_fp16 and "cuda" in self.config.device:
            model = model.half()
        model.eval()
        self.model = model

    def _mean_pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _to_chunk_item(self, chunk: ChunkItem | DBBgeM3Vector) -> ChunkItem:
        if isinstance(chunk, ChunkItem):
            return chunk
        return ChunkItem(
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
        )

    def _encode_chunk(self, chunk_item: ChunkItem) -> list[int]:
        return self.tokenizer.encode(
            self.PREFIX + chunk_item.content,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_length,
        )

    def embed_batch(self, batch: Sequence[str]) -> np.ndarray:
        texts = [self.PREFIX + text for text in batch]
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        
        return self._embed_encoded(encoded)

    def embed_token_ids_batch(self, batch_input_ids: list[list[int]]) -> np.ndarray:
        encoded = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding=True,
            return_tensors="pt",
        )
        return self._embed_encoded(encoded)

    def _embed_encoded(self, encoded: dict[str, torch.Tensor]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded; use TextEmbedding(config, load_model=True)")
        input_ids = encoded["input_ids"].to(self.config.device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(
            self.config.device, non_blocking=True
        )
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        return embeddings.detach().cpu().numpy()

    def token_worker_loop(self, chunk_queue: mp.Queue, token_queue: mp.Queue) -> None:
        while True:
            chunk_item = chunk_queue.get()
            if chunk_item is None:
                break
            token_queue.put((chunk_item, self._encode_chunk(chunk_item)))
        token_queue.put(None)

    def embedding_worker_loop(
        self,
        token_queue: mp.Queue,
        embedding_queue: mp.Queue,
        num_token_workers: int,
    ) -> None:
        batch: list[tuple[ChunkItem, list[int]]] = []
        workers_done = 0

        while workers_done < num_token_workers:
            item = token_queue.get()
            if item is None:
                workers_done += 1
                continue

            chunk_item, token_ids = item
            batch.append((chunk_item, token_ids))
            if len(batch) < self.config.batch_size:
                continue

            self._flush_embedding_batch(batch, embedding_queue)
            batch = []

        if batch:
            self._flush_embedding_batch(batch, embedding_queue)

        embedding_queue.put(None)

    def _flush_embedding_batch(
        self,
        batch: list[tuple[ChunkItem, list[int]]],
        embedding_queue: mp.Queue,
    ) -> None:
        token_id_batch = [token_ids for _, token_ids in batch]
        embeddings = self.embed_token_ids_batch(token_id_batch)
        for (chunk, _), vector in zip(batch, embeddings):
            embedding_queue.put((chunk, vector))

    def _drain_embedding_queue(
        self, embedding_queue: mp.Queue, sink: queue.Queue
    ) -> None:
        while True:
            item = embedding_queue.get()
            if item is None:
                sink.put(None)
                return
            sink.put(item)

    def run(
        self, chunklist: Sequence[ChunkItem | DBBgeM3Vector]
    ) -> Iterator[tuple[ChunkItem, np.ndarray]]:
        chunk_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        token_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        embedding_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)

        token_processes = [
            _MP_CTX.Process(
                target=_token_worker,
                args=(chunk_queue, token_queue, self.config),
            )
            for _ in range(self.config.num_workers)
        ]
        embedding_process = _MP_CTX.Process(
            target=_embedding_worker,
            args=(token_queue, embedding_queue, self.config, self.config.num_workers),
        )

        for process in token_processes:
            process.start()
        embedding_process.start()

        result_queue: queue.Queue = queue.Queue()
        drainer = threading.Thread(
            target=self._drain_embedding_queue,
            args=(embedding_queue, result_queue),
            daemon=True,
        )
        drainer.start()

        for chunk in tqdm(chunklist):
            chunk_queue.put(self._to_chunk_item(chunk))
        for _ in token_processes:
            chunk_queue.put(None)

        for process in token_processes:
            process.join()
        embedding_process.join()
        drainer.join()

        while True:
            try:
                item = result_queue.get_nowait()
            except Empty:
                break
            if item is None:
                break
            yield item
