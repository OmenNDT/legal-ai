import psycopg2
from ..data_pipeline.db_loader import DbConfig
from .retriever import RetrievedChunk

class InferenceLogger:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port,
            dbname = c.db, user = c.user, password = c.password
        )

    def log(self, user_question: str, chunks: list[RetrievedChunk], answer: str, latency_ms: int) -> None:
        chunk_ids = [c.chunk_id for c in chunks]
        sql = """
            INSERT INTO inference_logs (user_question, retrieved_chunk_ids, generated_answer, latency_ms)
            VALUES (%s, %s, %s, %s)
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (user_question, chunk_ids, answer, latency_ms))