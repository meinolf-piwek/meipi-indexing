from __future__ import annotations


from typing import Sequence

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

# from .text_chunking import PREFIX, prefix_token_ids
from ..model import ChunkItem
from ..config import EmbeddingConfig

class TextEmbedding:
    PREFIX = "passage: "
 

    def __init__(self, config: EmbeddingConfig, *, load_model: bool = True):
        self.config = config
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.model_name
        )
        self._prefix_token_ids = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self._max_content_tokens = config.max_length - len(self._prefix_token_ids)
        self.model: PreTrainedModel | None = None
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        if load_model:
            self._load_model()
    
    def _load_model(self) -> None:
        
        model = AutoModel.from_pretrained(self.config.model_name)
        model.to(self.device)  # type: ignore[arg-type]
        if self.config.use_fp16 and "cuda" in self.device:
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
        input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(
            self.device, non_blocking=True
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

    
    def embed_query(self, text: str) -> list[float]:
        if self.model is None:
            raise RuntimeError("Model not loaded; use TextEmbedding(config, load_model=True)")
        encoded = self.tokenizer.pad(
            {"input_ids": [self.encode_chunk_token_ids(text)]},
            padding=True,
            return_tensors="pt",
        )
        return self._embed_encoded(encoded)[0].tolist()