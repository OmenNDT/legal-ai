"""
Keyphrase extractor cho văn bản luật Việt Nam.

Dùng TF-IDF trên corpus từng bộ luật để tìm các cụm từ quan trọng nhất
trong mỗi chunk (Điều/Khoản/Điểm). Kết quả lưu vào bảng `keyphrases`.

Cách dùng:
    from chatbot.data_pipeline.keyphrase_extractor import KeyphraseExtractor, KeyphraseDbSaver
    from chatbot.data_pipeline.db_loader import DbConfig

    cfg = DbConfig.from_env()
    extractor = KeyphraseExtractor(top_k = 10, min_df = 2, ngram_max = 3)
    saver = KeyphraseDbSaver(cfg)
    extractor.run(cfg, saver, doc_code = "LKT2015")
"""

import re
import math
import psycopg2
from collections import Counter
from .db_loader import DbConfig

# Stopwords tiếng Việt cơ bản + các từ phổ biến trong văn bản luật nhưng ít ý nghĩa khi tách keyphrase.
_STOPWORDS = frozenset({
    "và", "của", "các", "có", "là", "được", "trong", "cho", "về",
    "theo", "tại", "đến", "từ", "với", "hoặc", "không", "này",
    "những", "đã", "sẽ", "để", "thì", "do", "khi", "trên", "dưới",
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "thứ", "đó", "như", "nếu", "mà", "vào", "ra", "lên", "xuống",
    "bởi", "vì", "nên", "nhưng", "còn", "sau", "trước", "ngoài",
    "qua", "khác", "cũng", "chỉ", "đây", "kể", "hơn", "ít", "nhiều",
    "phải", "được", "cần", "mọi", "tất", "cả", "một số", "quy định",
    "điều", "khoản", "điểm", "luật", "nghị", "định", "thông", "tư",
    "văn", "bản", "pháp", "luật", "số", "năm", "ngày", "tháng"
})

def _tokenize(text: str) -> list[str]:
    
    # Tách token đơn giản: lowercase, giữ dấu tiếng Việt, bỏ số đứng riêng.
    text = text.lower()
    text = re.sub(r"[^\wÀ-ỹ\s]", " ", text)
    tokens = [t for t in text.split() if t not in _STOPWORDS and not re.fullmatch(r"\d+", t) and len(t) > 1]
    return tokens

def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def _extract_ngram_candidates(tokens: list[str], max_n: int = 3) -> list[str]:
    # Tổng hợp unigram + bigram + trigram (nếu max_n >= 2/3).
    cands = tokens[:]
    for n in range(2, max_n + 1):
        cands.extend(_ngrams(tokens, n))
    return cands

class TfidfKeyphraseExtractor:
    """
    TF-IDF keyphrase extraction trên tập chunk của một văn bản luật.

    IDF được tính trên tập chunk trong cùng doc_code (không cross-document),
    để keyphrase phản ánh đặc thù của từng văn bản.
    """

    def __init__(self, top_k: int = 10, min_df: int = 2, ngram_max: int = 3) -> None:
        self._top_k = top_k
        self._min_df = min_df # Bỏ term xuất hiện < min_df chunk
        self._ngram_max = ngram_max

    def fit_transform(self, chunks: list[tuple[int, str]]) -> dict[int, list[str]]:
        """
        Tính keyphrase cho từng chunk.

        Args:
            chunks: list of (chunk_id, content_text)

        Returns:
            dict chunk_id → list of top keyphrase strings
        """
        # Bước 1: tokenize
        chunk_tokens: dict[int, list[str]] = {}
        for cid, text in chunks:
            tokens = _tokenize(text)
            cands = _extract_ngram_candidates(tokens, self._ngram_max)
            chunk_tokens[cid] = cands

        # Bước 2: document frequency
        df: Counter = Counter()
        for cands in chunk_tokens.values():
            for term in set(cands):
                df[term] += 1

        N = len(chunks)

        # Bước 3: TF-IDF per chunk
        result: dict[int, list[str]] = {}
        for cid, cands in chunk_tokens.items():
            tf = Counter(cands)
            total = sum(tf.values()) or 1
            scores: dict[str, float] = {}
            for term, count in tf.items():
                if df[term] < self._min_df:
                    continue
                idf = math.log((N + 1) / (df[term] + 1)) + 1.0
                scores[term] = (count / total) * idf

            # Lọc bỏ unigram nếu đã có ngram bao gồm nó với score cao hơn
            top = sorted(scores, key=lambda t: scores[t], reverse=True)
            deduped: list[str] = []
            seen_words: set[str] = set()
            for phrase in top:
                words = set(phrase.split())
                if words & seen_words:
                    continue
                deduped.append(phrase)
                seen_words.update(words)
                if len(deduped) >= self._top_k:
                    break
            result[cid] = deduped

        return result

class KeyphraseDbSaver:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port, dbname = c.db,
            user = c.user, password = c.password
        )

    def save(self, keyphrases_by_chunk: dict[int, list[str]]) -> None:
        sql = """
            INSERT INTO keyphrases (chunk_id, phrase, rank)
            VALUES (%s, %s, %s)
            ON CONFLICT (chunk_id, phrase) DO NOTHING
        """
        rows = [
            (cid, phrase, rank)
            for cid, phrases in keyphrases_by_chunk.items()
            for rank, phrase in enumerate(phrases, start=1)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
        print(f"[KeyphraseDbSaver] Đã lưu {len(rows)} keyphrases")

class KeyphraseExtractor:
    
    # Pipeline hoàn chỉnh: tải chunks từ DB → extract → lưu.

    def __init__(self, top_k: int = 10, min_df: int = 2, ngram_max: int = 3) -> None:
        self._inner = TfidfKeyphraseExtractor(top_k=top_k, min_df=min_df, ngram_max=ngram_max)

    def _connect(self, cfg: DbConfig):
        return psycopg2.connect(
            host = cfg.host, port = cfg.port, dbname = cfg.db,
            user = cfg.user, password = cfg.password
        )

    def _load_chunks(self, cfg: DbConfig, doc_code: str | None) -> list[tuple[int, str]]:
        if doc_code:
            sql = """
                SELECT lc.id, lc.content
                FROM law_chunks lc
                JOIN documents d ON d.id = lc.document_id
                WHERE (d.doc_code = %s OR d.short_code = %s) AND lc.content IS NOT NULL
                ORDER BY lc.id
            """
            params = (doc_code, doc_code)
        else:
            sql = "SELECT id, content FROM law_chunks WHERE content IS NOT NULL ORDER BY id"
            params = ()
        with self._connect(cfg) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def run(self, cfg: DbConfig, saver: KeyphraseDbSaver, doc_code: str | None = None) -> None:
        label = doc_code or "ALL"
        print(f"[KeyphraseExtractor] Tải chunks cho doc_code={label!r} ...")
        chunks = self._load_chunks(cfg, doc_code)
        print(f"[KeyphraseExtractor] Loaded {len(chunks)} chunks — đang extract ...")
        result = self._inner.fit_transform(chunks)
        saver.save(result)
        print(f"[KeyphraseExtractor] Hoàn tất — {sum(len(v) for v in result.values())} keyphrases tổng cộng")
