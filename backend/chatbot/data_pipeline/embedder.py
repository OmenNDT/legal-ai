import os
from typing import Sequence
from sentence_transformers import SentenceTransformer

class Embedder:
    # Wrapper nhúng text thành vector dùng BGE-M3
    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self._model = SentenceTransformer(name)

    def embed_one(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings = True)
        return vec.tolist()

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        vecs = self._model.encode(
            list(texts),
            batch_size = batch_size,
            normalize_embeddings = True,
            show_progress_bar = True
        )
        return [v.tolist() for v in vecs]
