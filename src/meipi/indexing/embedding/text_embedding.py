from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import time
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

from .text_preprocess import PREFIX, prefix_token_ids
from ..model import DBBgeM3Vector, ChunkItem
from ..operations import DBOperations

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
    def __init__(self, config: EmbeddingConfig, pool_id: int):
        self.config = config
        self.pool_id = pool_id
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
                chunk = ChunkItem(
                    doc_id=int(chunk.doc_id),
                    chunk_index=int(chunk.chunk_index),
                    content=str(chunk.content),
                )
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
            token_ids = embedder.encode_chunk_token_ids(chunk.content)
            token_queue.put((chunk, token_ids))
        token_queue.put(None)

       
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

    def dbwrite_worker(self,
        embedding_queue: mp.Queue,
    ) -> None:
        dbop = DBOperations(pool_id=self.pool_id)
        with dbop.Session() as session:
            for item in tqdm(iter(embedding_queue.get, None), desc="Writing chunks to database"):
                chunk, vector = item
                dbrow = session.get(DBBgeM3Vector, (chunk.doc_id, chunk.chunk_index)) 
                if dbrow is None:
                    dbrow = DBBgeM3Vector(doc_id=chunk.doc_id, chunk_index=chunk.chunk_index, 
                    content=chunk.content)
                    session.add(dbrow)
                dbrow.vector = vector
            session.flush()
            session.commit()
        

    def run_pipeline(self, chunklist: Sequence[ChunkItem | DBBgeM3Vector]) ->  None:
        start_time = time.time()
        chunk_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        token_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)
        embedding_queue: mp.Queue = _MP_CTX.Queue(maxsize=self.config.max_queue_size)

        print("Starting ingest process at", start_time)
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

        dbwrite_process = _MP_CTX.Process(
            target=self.dbwrite_worker,
            args=(embedding_queue,),
        )
        dbwrite_process.start()

        ingest_process.join()
        for process in token_processes:
            process.join()
        embedding_process.join()
        dbwrite_process.join()

        processes = [ingest_process, *token_processes, embedding_process, dbwrite_process]
        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            details = ", ".join(f"{p.name}(exitcode={p.exitcode})" for p in failed)
            raise RuntimeError(f"Embedding pipeline worker failed: {details}")

        end_time = time.time()
        print("End time:", end_time, "Start time:", start_time)
        print("Pipeline finished in", end_time - start_time, "seconds")


class TextEmbedding:
    

    def __init__(self, config: EmbeddingConfig, *, load_model: bool = True):
        self.config = config
        self.prefix = PREFIX
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.model_name
        )
        self._prefix_token_ids = prefix_token_ids(self.tokenizer)
        self._max_content_tokens = config.max_length - len(self._prefix_token_ids)
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

    
    def encode_chunk_token_ids(self, content: str) -> list[int]:
        content_ids = self.tokenizer.encode(
            content,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_content_tokens,
        )
        return self._prefix_token_ids + content_ids

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        token_ids = [self.encode_chunk_token_ids(text) for text in texts]
        return self.embed_token_ids(token_ids)

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

    