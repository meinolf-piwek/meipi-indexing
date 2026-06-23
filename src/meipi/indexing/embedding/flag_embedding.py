"""Speziell für BAAI Modelle entwickelt."""
from __future__ import annotations
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from ..config import EmbeddingConfig

class FlagEmbedding:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = BGEM3FlagModel(config.model_name, use_fp16=config.use_fp16, 
            device=self.device, batch_size=config.batch_size, passage_max_length=config.max_length,

        )

    
    def embed(self,texts: str|list[str]) -> np.ndarray:
        """Batch-Variante für viele Texte."""
        if isinstance(texts, str):
            texts = [texts]
        output = self.model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense_vecs: np.ndarray = output['dense_vecs'] # type: ignore[assignment] # type: ignore[type-arg]
        return dense_vecs