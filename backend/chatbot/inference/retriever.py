from dataclasses import dataclass
import psycopg2
from ..data_pipeline.db_loader import DbConfig
from ..data_pipeline.embedder import Embedder

@dataclass
class RetrievedChunk:
    chunk_id: int
    dieu: str
    khoan: str
    diem: str
    full_text: str
    similarity: float

class Retriever:
    def __init__(self, config: DbConfig, embedder: Embedder, top_k: int = 2, vector_weight: float = 0.7, trigram_weight: float = 0.3, rrf_k: int = 60) -> None:
        self._cfg = config
        self._embedder = embedder
        self._top_k = top_k
        self._vector_weight = vector_weight
        self._trigram_weight = trigram_weight
        self._rrf_k = rrf_k

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port,
            dbname = c.db, user = c.user, password = c.password
        )

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        vec = self._embedder.embed_one(question)
        vec_str = "[" + ",".join(map(str, vec)) + "]"
        k = self._top_k
        rrf_k = self._rrf_k
        vw = self._vector_weight
        tw = self._trigram_weight
        pool = max(k * 10, 20)

        sql = """
            WITH vector_ranked AS (
                SELECT 
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank,
                    1 - (embedding <=> %(vec)s::vector) AS vec_score
                FROM law_chunks
                ORDER BY embedding <=> %(vec)s::vector
                LIMIT %(pool)s
            ),
            trigram_ranked AS (
                SELECT 
                    id,
                    ROW_NUMBER() OVER (ORDER BY similarity(content, %(q)s) DESC) AS rank,
                    similarity(content, %(q)s) AS trgm_score
                FROM law_chunks
                WHERE content %% %(q)s
                ORDER BY similarity(content, %(q)s) DESC
                LIMIT %(pool)s
            ),
            rrf AS (
                SELECT
                    COALESCE(v.id, t.id) AS id,
                    COALESCE(%(vw)s / (%(rrf_k)s + v.rank), 0) +
                    COALESCE(%(tw)s / (%(rrf_k)s + t.rank), 0) AS rrf_score,
                    COALESCE(v.vec_score,  0) AS vec_score
                FROM vector_ranked v
                FULL OUTER JOIN trigram_ranked t ON v.id = t.id
            )
            SELECT lc.id, lc.dieu, lc.khoan, lc.diem, lc.full_text, r.vec_score
            FROM rrf r
            JOIN law_chunks lc ON lc.id = r.id
            ORDER BY r.rrf_score DESC
            LIMIT %(k)s
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, {
                "vec": vec_str,
                "q": question,
                "pool": pool,
                "vw": vw,
                "tw": tw,
                "rrf_k": rrf_k,
                "k": k
            })
            return [
                RetrievedChunk(
                    chunk_id = r[0],
                    dieu = r[1] or "",
                    khoan = r[2] or "",
                    diem = r[3] or "",
                    full_text = r[4] or "",
                    similarity = float(r[5])
                )
                for r in cur.fetchall()
            ]
