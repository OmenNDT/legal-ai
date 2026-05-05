import logging

logger = logging.getLogger(__name__)

class VectorStorageWriter:
    BATCH_SIZE = 256

    def __init__(self, vector_store):
        self._vector_store = vector_store

    def write(self, chunks_with_embeddings: list) -> list:
        logger.info("[VectorStore] Input: %d chunk(s)", len(chunks_with_embeddings))
        embedding_ids = []
        for i in range(0, len(chunks_with_embeddings), self.BATCH_SIZE):
            batch = chunks_with_embeddings[i: i + self.BATCH_SIZE]
            try:
                ids = self._write_batch(batch)
                embedding_ids.extend(ids)
                logger.info("[VectorStore] Upserted batch %d-%d (%d ids)", i, i + len(batch) - 1, len(ids))
            except Exception as e:
                logger.error("[VectorStore] Batch %d FAILED: %s", i, e, exc_info=True)
        logger.info("[VectorStore] Total upserted: %d", len(embedding_ids))
        return embedding_ids

    def _write_batch(self, batch: list) -> list:
        texts = [c["chunk_text"] for c in batch]
        embeddings = [c["embedding"] for c in batch]
        ids = [str(c["chunk_id"]) for c in batch]
        metadatas = [
            {
                "chunk_id": int(c["chunk_id"]),
                "document_title": c.get("document_title", ""),
                "document_type": c.get("document_type", ""),
                "article_number": c.get("article_number", ""),
                "clause_number": c.get("clause_number", ""),
                "point_number": c.get("point_number", ""),
                "effective_status": c.get("effective_status", "active"),
                "version_id": int(c.get("version_id", 0)),
                "section_id": int(c.get("section_id") or 0),
            }
            for c in batch
        ]
        self._vector_store._collection.upsert(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        return ids
