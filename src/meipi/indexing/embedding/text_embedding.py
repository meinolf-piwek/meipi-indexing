from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, cast

from tqdm.auto import tqdm
import multiprocessing as mp
import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)


from ..model import DBBgeM3Vector, ChunkItem

_MP_CTX = mp.get_context("spawn")





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

class EmbeddingPipeline:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.mp_ctx = mp.get_context("spawn")
    ###################
    # Worker functions #
    ###################
    def ingest_chunks_worker(self,
        chunk_list: Sequence[ChunkItem | DBBgeM3Vector],
        chunk_queue: mp.Queue,
    ) -> None:
        for chunk in tqdm(chunk_list, desc="Ingesting chunks"):
            if isinstance(chunk, DBBgeM3Vector):
                chunk = cast(ChunkItem, chunk)
            chunk_queue.put(chunk)
        for _ in range(self.config.num_workers):
            chunk_queue.put(None)
    
    def tokenize_chunks_worker(
        self,
        chunk_queue: mp.Queue,
        token_queue: mp.Queue,
    ) -> None:
        embedder = TextEmbedding(self.config, load_model=False)

        for chunk in tqdm(iter(chunk_queue.get,None), desc="Tokenizing chunks"):
            token_ids = embedder.tokenizer.encode(
                embedder.PREFIX + chunk.content,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.max_length,
            )
            token_queue.put((chunk, token_ids))
        token_queue.put(None)

        # while True:
        #     chunk_item = chunk_queue.get()
        #     if chunk_item is None:
        #         break
        #     token_ids = embedder.tokenizer.encode(
        #     embedder.PREFIX + chunk_item.content,
        #     add_special_tokens=False,
        #     truncation=True,
        #     max_length=self.config.max_length,
        # )
        #     token_queue.put((chunk_item, token_ids))
        # token_queue.put(None)


    def embedding_worker(self,
        token_queue: mp.Queue,
        embedding_queue: mp.Queue,
        num_token_workers: int,
    ) -> None:
        embedder = TextEmbedding(self.config, load_model=True)
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

            embedder._flush_embedding_batch(batch, embedding_queue)
            batch = []

        if batch:
            embedder._flush_embedding_batch(batch, embedding_queue)

        embedding_queue.put(None)

    def run_pipeline(self, chunklist: Sequence[ChunkItem | DBBgeM3Vector]) ->  Iterator[tuple[ChunkItem, np.ndarray]]:
        chunk_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        token_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        embedding_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)

        ingest_process = _MP_CTX.Process(
            target=self.ingest_chunks_worker,
            args=(chunklist, chunk_queue),
        )

        ingest_process.start()

        token_processes = [
            _MP_CTX.Process(
                target=self.tokenize_chunks_worker,
                args=(chunk_queue, token_queue),
            )
            for _ in range(self.config.num_workers)
        ]
        for process in token_processes:
            process.start()

        embedding_process = _MP_CTX.Process(
            target=self.embedding_worker,
            args=(token_queue, embedding_queue, len(token_processes)),
        )
        embedding_process.start()

        while True:
            item = embedding_queue.get()
            if item is None:
                break
            yield item
        ingest_process.join()
        for process in token_processes:
            process.join()
        embedding_process.join()



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

    ###################
    # Worker functions #
    ###################
    @classmethod
    def ingest_worker(cls,
        chunk_list: Sequence[ChunkItem | DBBgeM3Vector],
        chunk_queue: mp.Queue,
        config: EmbeddingConfig,
    ) -> None:
        for chunk in tqdm(chunk_list, desc="Ingesting chunks"):
            if isinstance(chunk, DBBgeM3Vector):
                chunk = cast(ChunkItem, chunk)
            chunk_queue.put(chunk)
        for _ in range(config.num_workers):
            chunk_queue.put(None)

    @classmethod
    def token_worker(
        cls,
        chunk_queue: mp.Queue,
        token_queue: mp.Queue,
        config: EmbeddingConfig,
    ) -> None:
        embedder = cls(config, load_model=False)
        while True:
            chunk_item = chunk_queue.get(timeout=10)
            if chunk_item is None:
                break
            token_ids = embedder.tokenizer.encode(
            embedder.PREFIX + chunk_item.content,
            add_special_tokens=False,
            truncation=True,
            max_length=embedder.config.max_length,
        )
            token_queue.put((chunk_item, token_ids))
        token_queue.put(None)


    @classmethod
    def embedding_worker(cls,
        token_queue: mp.Queue,
        embedding_queue: mp.Queue,
        config: EmbeddingConfig,
        num_token_workers: int,
    ) -> None:
        embedder = cls(config, load_model=True)
        batch: list[tuple[ChunkItem, list[int]]] = []
        workers_done = 0

        while workers_done < num_token_workers:
            item = token_queue.get()
            if item is None:
                workers_done += 1
                continue

            chunk_item, token_ids = item
            batch.append((chunk_item, token_ids))
            if len(batch) < config.batch_size:
                continue

            embedder._flush_embedding_batch(batch, embedding_queue)
            batch = []

        if batch:
            embedder._flush_embedding_batch(batch, embedding_queue)

        embedding_queue.put(None)




    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = [self.PREFIX + text for text in texts]
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            add_special_tokens=False,
        )     
        return self._embed_encoded(encoded)

    def embed_token_ids(self, input_ids: list[list[int]]) -> np.ndarray:
        encoded = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        return self._embed_encoded(encoded)

    def _embed_encoded(self, encoded: BatchEncoding) -> np.ndarray:
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
   
    def _flush_embedding_batch(
        self,
        batch: list[tuple[ChunkItem, list[int]]],
        embedding_queue: mp.Queue,
    ) -> None:
        token_id_batch = [token_ids for _, token_ids in batch]
        embeddings = self.embed_token_ids(token_id_batch)
        for (chunk, _), vector in zip(batch, embeddings):
            embedding_queue.put((chunk, vector))

    def run(
        self, chunklist: Sequence[ChunkItem | DBBgeM3Vector]
    ) -> Iterator[tuple[ChunkItem, np.ndarray]]:
        chunk_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        token_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        embedding_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)

        ingest_process = _MP_CTX.Process(
            target=self.ingest_worker,
            args=(chunklist, chunk_queue, self.config),
        )
        
        token_processes = [
            _MP_CTX.Process(
                target=self.token_worker,
                args=(chunk_queue, token_queue, self.config),
            )
            for _ in range(self.config.num_workers)
        ]
        print("Starting ingest process")
        ingest_process.start()
        print("Starting token processes")
        for process in token_processes:
            process.start()

        embedding_process = _MP_CTX.Process(
            target=self.embedding_worker,
            args=(token_queue, embedding_queue, self.config, len(token_processes)),
        )

        print("Starting embedding process")
        embedding_process.start()

        while True:
            item = embedding_queue.get()
            if item is None:
                break
            yield item
        ingest_process.join()
        for process in token_processes:
            process.join()
        embedding_process.join()
