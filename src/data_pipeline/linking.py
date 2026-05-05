import logging

logger = logging.getLogger(__name__)

class EmbeddingLinker:
    BATCH_SIZE = 500

    def __init__(self, db):
        self._db = db

    def link(self, chunk_ids: list) -> None:
        logger.info("[Link] Input: %d chunk_id(s)", len(chunk_ids))
        for i in range(0, len(chunk_ids), self.BATCH_SIZE):
            batch = chunk_ids[i: i + self.BATCH_SIZE]
            try:
                self._link_batch(batch)
                logger.info("[Link] Linked batch %d-%d", i, i + len(batch) - 1)
            except Exception as e:
                logger.error("[Link] Batch %d FAILED: %s", i, e, exc_info=True)
        logger.info("[Link] Done")

    def _link_batch(self, chunk_ids: list) -> None:
        params = [(str(cid), cid) for cid in chunk_ids]
        self._db.execute_many(
            "UPDATE legal_chunks SET embedding_id = %s WHERE id = %s AND embedding_id IS NULL",
            params,
        )
