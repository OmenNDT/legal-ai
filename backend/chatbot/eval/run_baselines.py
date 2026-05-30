import os
import json
import pickle
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any
import math
import numpy as np
import psycopg2
from dotenv import load_dotenv

HERE = Path(__file__).parent
# Find .env: walk up from this file, also try cwd
for p in [HERE / "..", HERE / "../..", HERE / "../../..", HERE / "../../../..", Path.cwd()]:
    env_path = (p / ".env").resolve()
    if env_path.is_file():
        load_dotenv(env_path)
        break

TEST_SET = HERE / "test_set.jsonl"
EMB_CACHE = HERE / "query_embeddings.pkl"

KS = [1, 3, 5, 10, 20]
MAX_K = max(KS)
POOL_SIZE = 50

def pg_connect():
    return psycopg2.connect(
        host = os.environ["POSTGRES_HOST"].strip(),
        port = int(os.environ["POSTGRES_PORT"].strip()),
        dbname = os.environ["POSTGRES_DB"].strip(),
        user = os.environ["POSTGRES_USER"].strip(),
        password = os.environ["POSTGRES_PASSWORD"].strip()
    )

def vec_to_pg(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

def retrieve_pure_vector(cur, vec_str: str, q: str, k: int) -> list[int]:
    cur.execute("""
        SELECT id FROM law_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (vec_str, k))
    return [r[0] for r in cur.fetchall()]

def retrieve_keyphrase(cur, q: str, k: int, sim_threshold: float = 0.3) -> list[int]:
    
    # Keyphrase-only: chunk có phrase khớp exact/fuzzy với query.
    cur.execute("""
        SELECT chunk_id
        FROM keyphrases
        WHERE phrase ILIKE %(ilike_q)s
           OR similarity(phrase, %(q)s) >= %(sim)s
        GROUP BY chunk_id
        ORDER BY MAX(similarity(phrase, %(q)s)) DESC
        LIMIT %(k)s
    """, {"q": q, "ilike_q": f"%{q}%", "sim": sim_threshold, "k": k})
    return [r[0] for r in cur.fetchall()]

# Replicates retriever.py (Hybrid RRF 3-source).
def retrieve_rrf(cur, vec_str: str, q: str, k: int, vw: float = 0.55, tw: float = 0.20, dw: float = 0.25, rrf_k: int = 60) -> list[int]:
    pool = max(k * 30, 200)
    cur.execute("""
        WITH vector_ranked AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank,
                   1 - (embedding <=> %(vec)s::vector) AS vec_score
            FROM law_chunks
            ORDER BY embedding <=> %(vec)s::vector LIMIT %(pool)s
        ),
        trigram_ranked AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY similarity(content, %(q)s) DESC) AS rank
            FROM law_chunks WHERE content %% %(q)s
            ORDER BY similarity(content, %(q)s) DESC LIMIT %(pool)s
        ),
        dieu_ranked AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY similarity(COALESCE(dieu,''), %(q)s) DESC) AS rank
            FROM law_chunks WHERE dieu IS NOT NULL AND similarity(dieu, %(q)s) > 0.15
            ORDER BY similarity(dieu, %(q)s) DESC LIMIT %(pool)s
        ),
        rrf AS (
            SELECT COALESCE(v.id, t.id, e.id) AS id,
                   COALESCE(%(vw)s / (%(rrf_k)s + v.rank), 0) +
                   COALESCE(%(tw)s / (%(rrf_k)s + t.rank), 0) +
                   COALESCE(%(dw)s / (%(rrf_k)s + e.rank), 0) AS rrf_score
            FROM vector_ranked v
            FULL OUTER JOIN trigram_ranked t ON v.id = t.id
            FULL OUTER JOIN dieu_ranked   e ON COALESCE(v.id, t.id) = e.id
        )
        SELECT lc.id,
               r.rrf_score + CASE WHEN COALESCE(lc.khoan,'') = '' AND COALESCE(lc.diem,'') = '' THEN 0.005 ELSE 0 END AS final_score
        FROM rrf r JOIN law_chunks lc ON lc.id = r.id
        ORDER BY final_score DESC LIMIT %(k)s
    """, {"vec": vec_str, "q": q, "pool": pool, "vw": vw, "tw": tw, "dw": dw, "rrf_k": rrf_k, "k": k})
    return [r[0] for r in cur.fetchall()]

def retrieve_rrf_keyphrase(cur, vec_str: str, q: str, k: int, vw: float = 0.50, tw: float = 0.18, dw: float = 0.22, kw: float = 0.10, rrf_k: int = 60) -> list[int]:
    pool = max(k * 30, 200)
    cur.execute("""
        WITH vector_ranked AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS rank
            FROM law_chunks ORDER BY embedding <=> %(vec)s::vector LIMIT %(pool)s
        ),
        trigram_ranked AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY similarity(content, %(q)s) DESC) AS rank
            FROM law_chunks WHERE content %% %(q)s
            ORDER BY similarity(content, %(q)s) DESC LIMIT %(pool)s
        ),
        dieu_ranked AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY similarity(COALESCE(dieu,''), %(q)s) DESC) AS rank
            FROM law_chunks WHERE dieu IS NOT NULL AND similarity(dieu, %(q)s) > 0.15
            ORDER BY similarity(dieu, %(q)s) DESC LIMIT %(pool)s
        ),
        keyphrase_ranked AS (
            SELECT chunk_id AS id,
                   ROW_NUMBER() OVER (ORDER BY MAX(similarity(phrase, %(q)s)) DESC) AS rank
            FROM keyphrases
            WHERE phrase ILIKE %(ilike_q)s OR similarity(phrase, %(q)s) >= 0.3
            GROUP BY chunk_id LIMIT %(pool)s
        ),
        rrf AS (
            SELECT COALESCE(v.id, t.id, e.id, kp.id) AS id,
                   COALESCE(%(vw)s / (%(rrf_k)s + v.rank), 0) +
                   COALESCE(%(tw)s / (%(rrf_k)s + t.rank), 0) +
                   COALESCE(%(dw)s / (%(rrf_k)s + e.rank), 0) +
                   COALESCE(%(kw)s / (%(rrf_k)s + kp.rank), 0) AS rrf_score
            FROM vector_ranked v
            FULL OUTER JOIN trigram_ranked t ON v.id = t.id
            FULL OUTER JOIN dieu_ranked e ON COALESCE(v.id, t.id) = e.id
            FULL OUTER JOIN keyphrase_ranked kp ON COALESCE(v.id, t.id, e.id) = kp.id
        )
        SELECT lc.id,
               r.rrf_score + CASE WHEN COALESCE(lc.khoan, '') = '' AND COALESCE(lc.diem,'') = '' THEN 0.005 ELSE 0 END AS final_score
        FROM rrf r JOIN law_chunks lc ON lc.id = r.id
        ORDER BY final_score DESC LIMIT %(k)s
    """, {"vec": vec_str, "q": q, "ilike_q": f"%{q}%", "pool": pool,
          "vw": vw, "tw": tw, "dw": dw, "kw": kw, "rrf_k": rrf_k, "k": k})
    return [r[0] for r in cur.fetchall()]


def reciprocal_rank(ranked: list[int], gold: int) -> float:
    try:
        return 1.0 / (ranked.index(gold) + 1)
    except ValueError:
        return 0.0

def ndcg_at_k(ranked: list[int], gold: int, k: int) -> float:
    for i, cid in enumerate(ranked[:k]):
        if cid == gold:
            return 1.0 / math.log2(i + 2)
    return 0.0

def answer_overlap(content: str, answer: str, min_words: int = 4) -> bool:
    if not content or not answer:
        return False
    a = " ".join(answer.lower().split())
    c = " ".join(content.lower().split())
    words = a.split()
    if len(words) < min_words:
        return a in c
    # Sliding window n-gram (n = min_words) — true nếu ít nhất 1 n-gram của answer xuất hiện trong content
    for i in range(len(words) - min_words + 1):
        ng = " ".join(words[i:i + min_words])
        if ng in c:
            return True
    return False

def measure_retriever(name: str, retriever_fn, queries, embeddings, cur, conn, fetch_contents_for_overlap=True):
    print(f"\n[{name}] running on {len(queries)} queries...")
    per_doc = defaultdict(lambda: {"n": 0, **{f"hit@{k}": 0 for k in KS}})
    summary: dict[str, Any] = {f"recall@{k}": 0 for k in KS}
    summary.update({f"mrr@{k}": 0.0 for k in KS})
    summary.update({f"ndcg@{k}": 0.0 for k in KS})
    summary["mean_rank_of_gold"] = 0.0
    summary["found_in_top"] = 0
    summary["overlap_in_top5"] = 0
    summary["overlap_in_top10"] = 0
    summary["latency_ms_avg"] = 0.0
    latencies = []

    rank_sum = 0.0
    rank_count = 0

    for i, q in enumerate(queries):
        vec = embeddings[i]
        vec_str = vec_to_pg(vec)
        t0 = time.perf_counter()
        ranked = retriever_fn(cur, vec_str, q["question"], MAX_K)
        latencies.append((time.perf_counter() - t0) * 1000)

        gold = q["gold_chunk_id"]
        if gold in ranked:
            r = ranked.index(gold) + 1
            rank_sum += r
            rank_count += 1
            summary["found_in_top"] += 1
        else:
            r = None

        for k in KS:
            hit = gold in ranked[:k]
            if hit:
                summary[f"recall@{k}"] += 1
                per_doc[q["doc_id"]][f"hit@{k}"] += 1
            if r is not None and r <= k:
                summary[f"mrr@{k}"] += 1.0 / r
                summary[f"ndcg@{k}"] += 1.0 / math.log2(r + 1)

        per_doc[q["doc_id"]]["n"] += 1

        if fetch_contents_for_overlap:
            top_ids = ranked[:10]
            if top_ids:
                cur.execute("SELECT id, content FROM law_chunks WHERE id = ANY(%s)", (top_ids,))
                id2content = {r[0]: r[1] or "" for r in cur.fetchall()}
                top5_contents = [id2content.get(c, "") for c in ranked[:5]]
                top10_contents = [id2content.get(c, "") for c in ranked[:10]]
                if any(answer_overlap(c, q["answer"]) for c in top5_contents):
                    summary["overlap_in_top5"] += 1
                if any(answer_overlap(c, q["answer"]) for c in top10_contents):
                    summary["overlap_in_top10"] += 1

        if (i + 1) % 50 == 0:
            print(f"[{name}] {i + 1}/{len(queries)} done")

    n = len(queries)
    for k in KS:
        summary[f"recall@{k}"] = round(summary[f"recall@{k}"] / n, 4)
        summary[f"mrr@{k}"] = round(summary[f"mrr@{k}"] / n, 4)
        summary[f"ndcg@{k}"] = round(summary[f"ndcg@{k}"] / n, 4)
    summary["mean_rank_of_gold"] = round(rank_sum / rank_count, 2) if rank_count else None
    summary["found_in_top"] = round(summary["found_in_top"] / n, 4)
    summary["overlap_in_top5"] = round(summary["overlap_in_top5"] / n, 4)
    summary["overlap_in_top10"] = round(summary["overlap_in_top10"] / n, 4)
    summary["latency_ms_p50"] = round(float(np.percentile(latencies, 50)), 1)
    summary["latency_ms_p95"] = round(float(np.percentile(latencies, 95)), 1)
    summary["latency_ms_avg"] = round(float(np.mean(latencies)), 1)

    per_doc_summary: dict[int, dict[str, Any]] = {}
    for d, stats in per_doc.items():
        row: dict[str, Any] = {"n": stats["n"]}
        for k in KS:
            row[f"recall@{k}"] = round(stats[f"hit@{k}"] / stats["n"], 4)
        per_doc_summary[int(d)] = row

    return {"summary": summary, "per_doc": per_doc_summary}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type = int, default = None, help = "Limit number of queries (debug)")
    p.add_argument("--retrievers", nargs = "+", default = ["vector", "rrf"], choices = ["vector", "rrf", "keyphrase", "rrf_keyphrase"])
    args = p.parse_args()

    rows = [json.loads(l) for l in TEST_SET.read_text(encoding = "utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[run] test set: {len(rows)} queries")

    with EMB_CACHE.open("rb") as f:
        cache = pickle.load(f)
    qa2vec = dict(zip(cache["qa_ids"], cache["vectors"]))
    embs = np.stack([qa2vec[r["qa_id"]] for r in rows])
    print(f"[run] embeddings ready: {embs.shape}")

    all_retrievers = [
        ("pure_vector", retrieve_pure_vector, ["vector", "pure_vector"]),
        ("hybrid_rrf", retrieve_rrf, ["rrf", "hybrid_rrf"]),
        ("keyphrase_only", lambda cur, vec_str, q, k: retrieve_keyphrase(cur, q, k), ["keyphrase", "keyphrase_only"]),
        ("hybrid_rrf_keyphrase", retrieve_rrf_keyphrase, ["rrf_keyphrase", "hybrid_rrf_keyphrase"])
    ]
    results = {}
    with pg_connect() as conn, conn.cursor() as cur:
        for name, fn, aliases in all_retrievers:
            if any(a in args.retrievers for a in aliases):
                results[name] = measure_retriever(name, fn, rows, embs, cur, conn)

    out = {
        "test_set_size": len(rows),
        "ks": KS,
        "retrievers": results,
        "doc_names": {}
    }

    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_code, doc_name FROM documents")
        for r in cur.fetchall():
            out["doc_names"][str(r[0])] = {"code": r[1], "name": r[2]}

    dst = HERE / "results_baseline.json"
    dst.write_text(json.dumps(out, ensure_ascii = False, indent = 2), encoding = "utf-8")
    print(f"\n[run] saved: {dst}")
    print(json.dumps({n: r["summary"] for n, r in results.items()}, indent = 2, ensure_ascii = False))

if __name__ == "__main__":
    main()
