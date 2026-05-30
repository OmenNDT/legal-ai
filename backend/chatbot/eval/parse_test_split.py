"""
Parse test.jsonl (generator eval set) thành retrieval test set:
- Mỗi row của test.jsonl có formatted_prompt + answer
- formatted_prompt = "Ngữ cảnh: {context} | Câu hỏi: {q} | Trả lời:"
- Join ngược vào training_samples qua formatted_prompt để lấy chunk_id_ref
- Sample stratified theo document_id để giảm xuống 1000 câu
"""
import os
import json
import random
import re
from pathlib import Path
from collections import defaultdict
import psycopg2
from dotenv import load_dotenv

HERE = Path(__file__).parent
for p in [HERE / "..", HERE / "../..", HERE / "../../..", HERE / "../../../..", Path.cwd()]:
    env_path = (p / ".env").resolve()
    if env_path.is_file():
        load_dotenv(env_path)
        break

random.seed(42)
TEST_SPLIT = HERE / ".." / "training" / "splits" / "test.jsonl"
OUT_PATH = HERE / "test_set_from_split.jsonl"
SAMPLE_SIZE = 1000

PROMPT_RE = re.compile(r"^Ngữ cảnh:\s*(.+?)\s*\|\s*Câu hỏi:\s*(.+?)\s*\|\s*Trả lời:\s*$", re.DOTALL)

def pg():
    return psycopg2.connect(
        host = os.environ["POSTGRES_HOST"].strip(),
        port = int(os.environ["POSTGRES_PORT"].strip()),
        dbname = os.environ["POSTGRES_DB"].strip(),
        user = os.environ["POSTGRES_USER"].strip(),
        password = os.environ["POSTGRES_PASSWORD"].strip()
    )

def main():
    rows = [json.loads(l) for l in TEST_SPLIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[parse] test.jsonl rows: {len(rows)}")

    parsed = []
    skipped = 0
    for r in rows:
        m = PROMPT_RE.match(r["prompt"])
        if not m:
            skipped += 1
            continue
        parsed.append({
            "context": m.group(1).strip(),
            "question": m.group(2).strip(),
            "answer": r["answer"],
            "prompt": r["prompt"]
        })
    print(f"[parse] parsed: {len(parsed)}, skipped (regex miss): {skipped}")

    # Map prompt → (qa_id_ref, chunk_id_ref, doc_id) qua training_samples
    print("[parse] mapping prompts to chunk_id via training_samples...")
    with pg() as conn, conn.cursor() as cur:
        prompts = [p["prompt"] for p in parsed]
        cur.execute("""
            SELECT ts.formatted_prompt, ts.qa_id_ref, ts.chunk_id_ref, ts.similarity_score,
                   lc.document_id, lc.dieu, d.doc_code, d.doc_name
            FROM training_samples ts
            JOIN law_chunks lc ON lc.id = ts.chunk_id_ref
            LEFT JOIN documents d ON d.id = lc.document_id
            WHERE ts.formatted_prompt = ANY(%s)
        """, (prompts,))
        # Mỗi formatted_prompt có thể có nhiều chunk_id (top-2 mapping per qa) — giữ tất cả
        prompt_to_chunks: dict[str, list] = defaultdict(list)
        for row in cur.fetchall():
            prompt_to_chunks[row[0]].append({
                "qa_id": int(row[1]),
                "chunk_id": int(row[2]),
                "similarity": float(row[3]),
                "doc_id": row[4],
                "dieu": row[5] or "",
                "doc_code": row[6] or "",
                "doc_name": row[7] or ""
            })

    enriched = []
    for p in parsed:
        chunks = prompt_to_chunks.get(p["prompt"], [])
        if not chunks:
            continue
        best = max(chunks, key = lambda c: c["similarity"])
        enriched.append({**p, "weak_gold_chunk_id": best["chunk_id"], "qa_id": best["qa_id"],
            "weak_gold_similarity": best["similarity"], "doc_id": best["doc_id"],
            "doc_code": best["doc_code"], "doc_name": best["doc_name"],
            "all_mapped_chunks": [c["chunk_id"] for c in chunks]})
    print(f"[parse] enriched (with chunk mapping): {len(enriched)}")

    # Stratified sample by doc_id
    by_doc = defaultdict(list)
    for e in enriched:
        by_doc[e["doc_id"]].append(e)
    total = len(enriched)
    sampled = []
    for _, items in by_doc.items():
        n = max(1, round(SAMPLE_SIZE * len(items) / total))
        random.shuffle(items)
        sampled.extend(items[:n])
    if len(sampled) > SAMPLE_SIZE:
        random.shuffle(sampled)
        sampled = sampled[:SAMPLE_SIZE]
    print(f"[parse] stratified sample: {len(sampled)}")
    print(f"[parse] doc distribution:")
    doc_count: dict = defaultdict(int)
    for s in sampled: doc_count[s["doc_code"]] += 1
    for code, n in sorted(doc_count.items(), key = lambda x: -x[1]):
        print(f"{code}: {n}")

    OUT_PATH.write_text("\n".join(json.dumps(s, ensure_ascii = False) for s in sampled), encoding = "utf-8")
    print(f"[parse] saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
