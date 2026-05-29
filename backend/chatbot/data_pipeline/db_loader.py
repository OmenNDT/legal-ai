import os
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import execute_values
from .law_parser import LawChunk

@dataclass
class DbConfig:
    host: str
    port: int
    db: str
    user: str
    password: str
    @classmethod
    def from_env(cls) -> "DbConfig":
        return cls(
            host = os.environ["POSTGRES_HOST"],
            port = int(os.environ.get("POSTGRES_PORT", 5432)),
            db = os.environ["POSTGRES_DB"],
            user = os.environ["POSTGRES_USER"],
            password = os.environ["POSTGRES_PASSWORD"]
        )
class DbLoader:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config
    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host,
            port = c.port,
            dbname = c.db,
            user = c.user,
            password = c.password
        )
    def upsert_document(self, doc_code: str, doc_name: str, doc_type: str, issue_year: int | None) -> int:
        sql = """
            INSERT INTO documents (doc_code, doc_name, doc_type, issue_year)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_code) DO UPDATE SET doc_name = EXCLUDED.doc_name
            RETURNING id
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (doc_code, doc_name, doc_type, issue_year))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"upsert_document không trả về id cho doc_code = {doc_code!r}")
            return row[0]

    def load_chunks(self, document_id: int, chunks: list[LawChunk], embeddings: list[list[float]]) -> None:
        sql = """
            INSERT INTO law_chunks (document_id, hierarchy_path, phan, chuong, muc, dieu, khoan, diem, content, full_text, embedding)
            VALUES %s
        """
        rows = [
            (
                document_id,
                chunk.hierarchy_path,
                chunk.phan,
                chunk.chuong,
                chunk.muc,
                chunk.dieu,
                chunk.khoan,
                chunk.diem,
                chunk.content,
                chunk.full_text,
                embedding
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            execute_values(cur, sql, rows)
        print(f"[DbLoader] Đã nạp {len(rows)} chunks cho document_id = {document_id}")
