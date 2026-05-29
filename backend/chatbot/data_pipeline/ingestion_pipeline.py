import argparse
import re
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from .pdf_extractor import PdfExtractor
from .law_parser import HierarchicalParser, LawChunk
from .embedder import Embedder
from .db_loader import DbLoader, DbConfig

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

class DocMetaRegistry:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config
        self._cache: dict[str, tuple[str, str, str, int | None]] = {}

    def _load(self) -> None:
        if self._cache:
            return
        c = self._cfg
        with psycopg2.connect(host = c.host, port = c.port, dbname = c.db, user = c.user, password = c.password) as conn, conn.cursor() as cur:
            cur.execute("SELECT doc_code, doc_name, doc_type, issue_year FROM documents")
            for doc_code, doc_name, doc_type, issue_year in cur.fetchall():
                key = re.sub(r"[\s\-]+", "_", doc_name.lower())
                self._cache[key] = (doc_code, doc_name, doc_type, issue_year)

    def resolve(self, pdf_stem: str) -> tuple[str, str, str, int | None]:
        self._load()
        norm = pdf_stem.lower()
        for key, meta in self._cache.items():
            if norm in key or key in norm:
                return meta
        doc_name = pdf_stem.replace("_", " ")
        year_match = re.search(r"(\d{4})", pdf_stem)
        year = int(year_match.group(1)) if year_match else None
        return (pdf_stem, doc_name, "Luật", year)

class IngestionPipeline:
    def __init__(self, extractor: PdfExtractor, embedder: Embedder, loader: DbLoader, registry: DocMetaRegistry, batch_size: int = 64) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._loader = loader
        self._registry = registry
        self._batch_size = batch_size

    def run(self, pdf_path: Path) -> None:
        print(f"[Ingestion] Xử lý: {pdf_path.name}")
        doc_code, doc_name, doc_type, issue_year = self._registry.resolve(pdf_path.stem)

        # Bước 1: Extract + clean text
        full_text = self._extractor.extract_full_text(pdf_path)

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

        print(f"[Ingestion] Hoàn tất {pdf_path.name} — {len(chunks)} chunks nạp vào DB")

    def run_directory(self, data_dir: Path) -> None:
        # Xử lý tất cả PDF trong thư mục (đệ quy)
        pdfs = list(data_dir.rglob("*.pdf"))
        print(f"[Ingestion] Tìm thấy {len(pdfs)} file PDF")
        for pdf in pdfs:
            self.run(pdf)

def _build_pipeline() -> IngestionPipeline:
    cfg = DbConfig.from_env()
    return IngestionPipeline(
        extractor = PdfExtractor(),
        embedder = Embedder(),
        loader = DbLoader(cfg),
        registry = DocMetaRegistry(cfg)
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Phase 1 — PDF Ingestion Pipeline")
    group  = parser.add_mutually_exclusive_group(required = True)
    group.add_argument("--pdf", type = Path, help = "Đường dẫn đến một file PDF")
    group.add_argument("--dir", type = Path, help = "Thư mục chứa các file PDF")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline = _build_pipeline()
    if args.pdf:
        pipeline.run(args.pdf)
    else:
        pipeline.run_directory(args.dir)