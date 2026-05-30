import os
import json
import random
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
random.seed(42)

def conn():
    return psycopg2.connect(
        host = os.environ["POSTGRES_HOST"].strip(),
        port = int(os.environ["POSTGRES_PORT"].strip()),
        dbname = os.environ["POSTGRES_DB"].strip(),
        user = os.environ["POSTGRES_USER"].strip(),
        password = os.environ["POSTGRES_PASSWORD"].strip()
    )

SQL = """
    WITH best AS (
        SELECT DISTINCT ON (ts.qa_id_ref)
            ts.qa_id_ref, ts.chunk_id_ref, ts.similarity_score
        FROM training_samples ts
        WHERE ts.similarity_score >= 0.7
        ORDER BY ts.qa_id_ref, ts.similarity_score DESC
    )
    SELECT
        b.qa_id_ref, b.chunk_id_ref, b.similarity_score,
        q.question, q.answer,
        lc.document_id, lc.dieu, lc.khoan, lc.diem, lc.full_text,
        d.doc_code, d.doc_name
    FROM best b
    JOIN qa_data q ON q.id = b.qa_id_ref
    JOIN law_chunks lc ON lc.id = b.chunk_id_ref
    LEFT JOIN documents d ON d.id = lc.document_id
    WHERE q.question IS NOT NULL AND q.answer IS NOT NULL AND LENGTH(q.question) >= 10 AND LENGTH(q.answer) >= 10
"""

def main(test_size: int = 500):
    with conn() as c, c.cursor() as cur:
        cur.execute(SQL)
        rows = cur.fetchall()

    print(f"[build] Eligible QA (sim ≥ 0.7): {len(rows)}")
    by_doc: dict[int, list] = {}
    for r in rows:
        by_doc.setdefault(r[5], []).append(r)
    print(f"[build] Docs covered: {len(by_doc)}")
    for d, items in sorted(by_doc.items(), key = lambda x: -len(x[1])):
        print(f"doc_id = {d} count = {len(items)}")

    total = len(rows)
    sampled = []
    for d, items in by_doc.items():
        n = max(1, round(test_size * len(items) / total))
        random.shuffle(items)
        sampled.extend(items[:n])
    if len(sampled) > test_size:
        random.shuffle(sampled)
        sampled = sampled[:test_size]
    print(f"[build] Test set size: {len(sampled)}")

    out = []
    for r in sampled:
        out.append({
            "qa_id": int(r[0]),
            "gold_chunk_id": int(r[1]),
            "gold_similarity": float(r[2]),
            "question": r[3],
            "answer": r[4],
            "doc_id": r[5],
            "dieu": r[6] or "",
            "khoan": r[7] or "",
            "diem": r[8] or "",
            "gold_full_text": r[9] or "",
            "doc_code": r[10] or "",
            "doc_name": r[11] or ""
        })

    dst = Path(__file__).parent / "test_set.jsonl"
    dst.write_text("\n".join(json.dumps(o, ensure_ascii = False) for o in out), encoding = "utf-8")
    print(f"[build] Wrote: {dst}")

if __name__ == "__main__":
    main()
