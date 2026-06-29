import re

from transformers import AutoTokenizer, PreTrainedTokenizer
from itertools import chain

from ..model import DocItem, ChunkItem
from ..text_cleaning import clean_document_text
from ..config import EmbeddingConfig


class DocumentChunker:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._space_tokens = self.tokenizer.encode(" ", add_special_tokens=False)

    def chunk_doc(self, doc: DocItem) -> list[ChunkItem]:
        if doc.inhalt is None or doc.inhalt == "":
            return []
        if self.config.clean_text:
            doc.inhalt = self._clean_text(doc.inhalt)

        paragraphs = self._split_paragraphs(doc.inhalt)
        chunks = self._build_chunks(paragraphs)
        return [ChunkItem(doc_id=doc.id, chunk_index=index, content=chunk) for index, chunk in enumerate(chunks)]

    def chunk_doclist(self, docs: list[DocItem]) -> list[ChunkItem]:
        return list(chain.from_iterable(self.chunk_doc(doc) for doc in docs))

    def _clean_text(self, text: str) -> str:
        return clean_document_text(text)

    def _split_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _build_chunks(self, paragraphs: list[str]) -> list[str]:
        if not paragraphs:
            return []

        para_token_lists = self._encode_paragraphs(paragraphs)
        chunk_token_lists = self._merge_paragraph_tokens(para_token_lists)
        chunk_token_lists = self._apply_overlap_tokens(chunk_token_lists)
        chunks: list[str] = []
        for tokenlist in chunk_token_lists:
            chunks.append(self.tokenizer.decode(tokenlist, skip_special_tokens=True))  # type: ignore[arg-type]
        return chunks

    def _encode_paragraphs(self, paragraphs: list[str]) -> list[list[int]]:
        if len(paragraphs) == 1:
            return [self.tokenizer.encode(paragraphs[0], add_special_tokens=False)]
        return self.tokenizer(paragraphs, add_special_tokens=False)["input_ids"]

    def _merge_paragraph_tokens(
        self, para_token_lists: list[list[int]]
    ) -> list[list[int]]:
        chunks: list[list[int]] = []
        current: list[int] = []
        current_len = 0

        for para_tokens in para_token_lists:
            para_len = len(para_tokens)
            if para_len > self.config.max_length:
                if current:
                    chunks.append(current)
                    current = []
                    current_len = 0
                chunks.extend(self._split_long_tokens(para_tokens))
                continue

            extra = para_len + (len(self._space_tokens) if current else 0)
            if current_len + extra <= self.config.max_length:
                if current:
                    current.extend(self._space_tokens)
                current.extend(para_tokens)
                current_len += extra
            else:
                if current:
                    chunks.append(current)
                current = list(para_tokens)
                current_len = para_len

        if current:
            chunks.append(current)
        return chunks

    def _split_long_tokens(self, tokens: list[int]) -> list[list[int]]:
        step = self.config.max_length - self.config.overlap
        chunks: list[list[int]] = []
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.config.max_length]
            if len(chunk_tokens) >= self.config.min_chunk_tokens:
                chunks.append(chunk_tokens)
        return chunks

    def _apply_overlap_tokens(self, chunks: list[list[int]]) -> list[list[int]]:
        if self.config.overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_overlap = chunks[i - 1][-self.config.overlap :]
            merged = prev_overlap + self._space_tokens + chunks[i]
            overlapped.append(merged)
        return overlapped
