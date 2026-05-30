import argparse
import re
import unicodedata
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from .txt_extractor import TxtExtractor
from .law_parser import HierarchicalParser, LawChunk
from .embedder import Embedder
from .db_loader import DbLoader, DbConfig

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

def _remove_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

class DocMetaRegistry:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config
        # key = tên DB đã bỏ dấu + lowercase + underscore
        self._cache: dict[str, tuple[str, str, str, int | None]] = {}

    def _load(self) -> None:
        if self._cache:
            return
        c = self._cfg
        with psycopg2.connect(host = c.host, port = c.port, dbname = c.db, user = c.user, password = c.password) as conn, conn.cursor() as cur:
            cur.execute("SELECT doc_code, doc_name, doc_type, issue_year FROM documents")
            for doc_code, doc_name, doc_type, issue_year in cur.fetchall():
                # Index bằng tên không dấu, chữ thường, khoảng trắng/gạch → "_"
                key = re.sub(r"[\s\-]+", "_", _remove_accents(doc_name).lower())
                self._cache[key] = (doc_code, doc_name, doc_type, issue_year)

    def resolve(self, txt_stem: str) -> tuple[str, str, str, int | None]:
        self._load()
        # Normalize tên file giống key trong cache
        norm = re.sub(r"[\s\-]+", "_", _remove_accents(txt_stem).lower())
        # Trích năm từ tên file để tránh nhầm khi có 2 phiên bản (vd Đất đai 2013 vs 2024)
        year_match = re.search(r"(\d{4})", txt_stem)
        year_str = year_match.group(1) if year_match else ""
        for key, meta in self._cache.items():
            if year_str and year_str not in key:
                continue
            if norm in key or key in norm:
                return meta
        # Fallback: không tìm thấy trong DB — không nên xảy ra nếu documents đã được seed
        doc_name = txt_stem.replace("_", " ")
        year = int(year_str) if year_str else None
        print(f"[DocMetaRegistry] WARN: không tìm thấy '{txt_stem}' trong bảng documents, tạo fallback")
        return (txt_stem, doc_name, "Luật", year)

class IngestionPipeline:
    def __init__(self, extractor: TxtExtractor, embedder: Embedder, loader: DbLoader, registry: DocMetaRegistry, batch_size: int = 64) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._loader = loader
        self._registry = registry
        self._batch_size = batch_size

    def run(self, txt_path: Path) -> None:
        print(f"[Ingestion] Xử lý: {txt_path.name}")
        doc_code, doc_name, doc_type, issue_year = self._registry.resolve(txt_path.stem)

        # Bước 1: Extract + clean text
        full_text = self._extractor.extract_full_text(txt_path)

        # Bước 2: Bóc tách phân cấp
        parser = HierarchicalParser(doc_name)
        chunks: list[LawChunk] = list(parser.parse(full_text))
        print(f"[Ingestion] Bóc tách được {len(chunks)} chunks")

        # Bước 3: Upsert document record
        document_id = self._loader.upsert_document(doc_code, doc_name, doc_type, issue_year)

        # Bước 4: Embed theo batch rồi load
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            texts = [c.full_text for c in batch]
            vectors = self._embedder.embed_batch(texts)
            self._loader.load_chunks(document_id, batch, vectors)

        print(f"[Ingestion] Hoàn tất {txt_path.name} — {len(chunks)} chunks nạp vào DB")

    def run_directory(self, data_dir: Path) -> None:
        # Xử lý tất cả TXT trong thư mục (đệ quy)
        txts = list(data_dir.rglob("*.txt"))
        print(f"[Ingestion] Tìm thấy {len(txts)} file TXT")
        for txt in txts:
            self.run(txt)

def _build_pipeline() -> IngestionPipeline:
    cfg = DbConfig.from_env()
    return IngestionPipeline(
        extractor = TxtExtractor(),
        embedder = Embedder(),
        loader = DbLoader(cfg),
        registry = DocMetaRegistry(cfg)
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Phase 1 — TXT Ingestion Pipeline")
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument("--txt", type = Path, help = "Đường dẫn đến một file TXT")
    group.add_argument("--dir", type = Path, help = "Thư mục chứa các file TXT")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline = _build_pipeline()
    if args.txt:
        pipeline.run(args.txt)
    else:
        pipeline.run_directory(args.dir)