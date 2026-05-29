from dataclasses import dataclass
import psycopg2
import psycopg2.extras
from ..data_pipeline.embedder import Embedder
from ..data_pipeline.db_loader import DbConfig

@dataclass
class QaRow:
    qa_id: int
    question: str
    answer: str

@dataclass
class MappedSample:
    qa_id: int
    chunk_id: int
    similarity_score: float
    question: str
    context: str
    answer: str

    @property
    def formatted_prompt(self) -> str:
        return f"Ngữ cảnh: {self.context} | Câu hỏi: {self.question} | Trả lời:"

    @property
    def full_training_text(self) -> str:
        return f"Ngữ cảnh: {self.context} | Câu hỏi: {self.question} | Trả lời: {self.answer}"

class QaLoader:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port,
            dbname = c.db, user = c.user, password = c.password
        )

    def load(self, limit: int | None = None) -> list[QaRow]:
        sql = """
            SELECT COALESCE(id, idx), question, answer
            FROM qa_data
            WHERE question IS NOT NULL AND answer IS NOT NULL AND LENGTH(question) > 0 AND LENGTH(answer) > 0
        """
        if limit:
            sql += f" LIMIT {limit}"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql)
            return [QaRow(qa_id = r[0], question = r[1], answer = r[2]) for r in cur.fetchall()]

class VectorSearcher:
    def __init__(self, config: DbConfig, top_k: int = 2) -> None:
        self._cfg = config
        self._top_k = top_k

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port,
            dbname = c.db, user = c.user, password = c.password
        )

    def search(self, query_vec: list[float]) -> list[tuple[int, str, float]]:
        sql = """
            SELECT 
                id, 
                full_text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM law_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vec_str = "[" + ",".join(map(str, query_vec)) + "]"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, (vec_str, vec_str, self._top_k))
            return [(r[0], r[1], float(r[2])) for r in cur.fetchall()]

class TrainingSampleSaver:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port,
            dbname = c.db, user = c.user, password = c.password
        )

    def save_batch(self, samples: list[MappedSample]) -> None:
        sql = """
            INSERT INTO training_samples (qa_id_ref, chunk_id_ref, similarity_score, formatted_prompt, target_answer)
            VALUES %s
        """
        rows = [
            (s.qa_id, s.chunk_id, s.similarity_score, s.formatted_prompt, s.answer)
            for s in samples
        ]
        with self._connect() as conn, conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows)
        print(f"[TrainingSampleSaver] Đã lưu {len(rows)} samples")

class QaMapper:
    def __init__(self, qa_loader: QaLoader, embedder: Embedder, searcher: VectorSearcher, saver: TrainingSampleSaver, min_similarity: float = 0.5, batch_size: int = 64) -> None:
        self._qa_loader = qa_loader
        self._embedder = embedder
        self._searcher = searcher
        self._saver = saver
        self._min_similarity = min_similarity
        self._batch_size = batch_size

    def run(self, limit: int | None = None) -> None:
        qa_rows = self._qa_loader.load(limit = limit)
        print(f"[QaMapper] Loaded {len(qa_rows)} Q&A rows")

        mapped: list[MappedSample] = []

        for i, row in enumerate(qa_rows):
            vec = self._embedder.embed_one(row.question)
            results = self._searcher.search(vec)

            for chunk_id, full_text, score in results:
                if score < self._min_similarity:
                    continue
                mapped.append(MappedSample(
                    qa_id = row.qa_id,
                    chunk_id = chunk_id,
                    similarity_score = score,
                    question = row.question,
                    context = full_text,
                    answer = row.answer
                ))
            if len(mapped) >= self._batch_size:
                self._saver.save_batch(mapped)
                mapped = []
            if (i + 1) % 100 == 0:
                print(f"[QaMapper] Đã xử lý {i + 1}/{len(qa_rows)}")
        if mapped:
            self._saver.save_batch(mapped)
        print("[QaMapper] Hoàn tất mapping Q&A → training_samples")