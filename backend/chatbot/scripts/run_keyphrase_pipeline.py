"""
Chạy full pipeline: migrate → keyphrase extract → concept tag → QA generate
cho một văn bản luật cụ thể.

Cách dùng:
    python3 backend/chatbot/scripts/run_keyphrase_pipeline.py --doc LKT2015
    python3 backend/chatbot/scripts/run_keyphrase_pipeline.py --doc ALL
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
d = dotenv_values(ROOT / ".env")

from backend.chatbot.data_pipeline.db_loader import DbConfig
import psycopg2

cfg = DbConfig(
    host=d["POSTGRES_HOST"], port=int(d.get("POSTGRES_PORT", 5432)),
    db=d["POSTGRES_DB"], user=d["POSTGRES_USER"], password=d["POSTGRES_PASSWORD"]
)

def get_conn():
    return psycopg2.connect(
        host=d["POSTGRES_HOST"], port=int(d.get("POSTGRES_PORT", 5432)),
        dbname=d["POSTGRES_DB"], user=d["POSTGRES_USER"], password=d["POSTGRES_PASSWORD"]
    )

def apply_migration():
    print("=== Migration ===")
    sql = (ROOT / "backend/chatbot/migrations/002_keyphrases_concepts.sql").read_text()
    conn = get_conn()
    conn.autocommit = True
    conn.cursor().execute(sql)
    conn.close()
    print("[OK]")

def run_keyphrase(doc_code):
    print(f"\n=== KeyphraseExtractor ({doc_code or 'ALL'}) ===")
    from backend.chatbot.data_pipeline.keyphrase_extractor import KeyphraseExtractor, KeyphraseDbSaver
    KeyphraseExtractor(top_k=10, min_df=2, ngram_max=3).run(cfg, KeyphraseDbSaver(cfg), doc_code=doc_code)

def run_concept(doc_code):
    print(f"\n=== ConceptTagger ({doc_code or 'ALL'}) ===")
    from backend.chatbot.data_pipeline.concept_tagger import ConceptTagger, ConceptDbSaver
    ConceptTagger().run(cfg, ConceptDbSaver(cfg), doc_code=doc_code)

def run_qa(doc_code):
    if not doc_code:
        print("\n[SKIP] QA generator chỉ hỗ trợ từng doc_code cụ thể")
        return
    print(f"\n=== KeToaQaGenerator ({doc_code}) ===")
    from backend.chatbot.training.ke_toan_qa_generator import KeToaQaGenerator
    KeToaQaGenerator(cfg).run(doc_code=doc_code)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="LKT2015", help="doc_code hoặc short_code, hoặc ALL")
    ap.add_argument("--skip-migration", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()

    doc = None if args.doc == "ALL" else args.doc

    if not args.skip_migration:
        apply_migration()
    run_keyphrase(doc)
    run_concept(doc)
    if not args.skip_qa:
        run_qa(doc)

    print("\n=== DONE ===")
