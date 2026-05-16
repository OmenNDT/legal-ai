import logging
import os
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    BATCH_SIZE = 64

    def __init__(self, model_name: str):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        self._model = SentenceTransformer(model_name, device="cpu")
        logger.info("[Embed] Model loaded on CPU: %s", model_name)

    def generate(self, chunks: list) -> list:
        logger.info("[Embed] Input: %d chunk(s)", len(chunks))
        texts = [
            chunk.get("segmented_text") or chunk.get("chunk_text", "")
            for chunk in chunks
        ]
        embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i: i + self.BATCH_SIZE]
            try:
                batch_embeddings = self._model.encode(batch, show_progress_bar=False).tolist()
                embeddings.extend(batch_embeddings)
                logger.info("[Embed] Batch %d-%d done", i, i + len(batch) - 1)
            except Exception as e:
                logger.error("[Embed] Batch %d FAILED: %s", i, e, exc_info=True)
                embeddings.extend([None] * len(batch))
        result = [
            {**chunk, "embedding": embedding}
            for chunk, embedding in zip(chunks, embeddings)
            if embedding is not None
        ]
        logger.info("[Embed] Done: %d/%d chunk(s) embedded", len(result), len(chunks))
        return result
