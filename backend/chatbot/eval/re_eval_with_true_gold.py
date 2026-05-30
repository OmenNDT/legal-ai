"""
Re-evaluate 3 retrievers (pure_vector, hybrid_rrf, vector+reranker) trên test set 999
với TRUE gold labels do Qwen judge sinh ra.

Multi-gold metrics:
- Recall@K = (|gold ∩ top-K|) / |gold|  — bao nhiêu phần trăm gold lọt vào top-K
- Hit@K = 1 nếu ít nhất 1 gold trong top-K, else 0
- MRR@K = 1 / rank của gold đầu tiên trong top-K
- NDCG@K = graded relevance (score 2 = highly relevant, 1 = related)

Output: results_with_true_gold.json — merge vào results_baseline.json hoặc thay thế.
"""
import os
import json
import time
import argparse
import math
from pathlib import Path
from collections import defaultdict
from typing import Any
import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

HERE = Path(__file__).parent
for p in [HERE / "..", HERE / "../..", HERE / "../../..", HERE / "../../../..", Path.cwd()]:
    env_path = (p / ".env").resolve()
    if env_path.is_file():
        load_dotenv(env_path)
        break

GOLD_PATH = HERE / "gold_labels_qwen.jsonl"
OUT_PATH = HERE / "results_baseline.json"
KS = [1, 3, 5, 10, 20]
MAX_K = max(KS)

def pg():
    return psycopg2.connect(
        host = os.environ["POSTGRES_HOST"].strip(),
        port = int(os.environ["POSTGRES_PORT"].strip()),
        dbname = os.environ["POSTGRES_DB"].strip(),
        user = os.environ["POSTGRES_USER"].strip(),
        password = os.environ["POSTGRES_PASSWORD"].strip()
    )

def vec_pg(v): return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

def retrieve_pure_vector(cur, vec_str, q, k):
    cur.execute("SELECT id FROM law_chunks ORDER BY embedding <=> %s::vector LIMIT %s", (vec_str, k))
    return [r[0] for r in cur.fetchall()]

def retrieve_keyphrase(cur, q, k, sim_threshold=0.3):
    cur.execute("""
        SELECT chunk_id FROM keyphrases
        WHERE phrase ILIKE %(ilike_q)s OR similarity(phrase, %(q)s) >= %(sim)s
        GROUP BY chunk_id
        ORDER BY MAX(similarity(phrase, %(q)s)) DESC
        LIMIT %(k)s
    """, {"q": q, "ilike_q": f"%{q}%", "sim": sim_threshold, "k": k})
    return [r[0] for r in cur.fetchall()]

def retrieve_rrf_keyphrase(cur, vec_str, q, k, vw = 0.50, tw = 0.18, dw = 0.22, kw = 0.10, rrf_k = 60):
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
                   COALESCE(%(vw)s / (%(rrf_k)s + v.rank),  0) +
                   COALESCE(%(tw)s / (%(rrf_k)s + t.rank),  0) +
                   COALESCE(%(dw)s / (%(rrf_k)s + e.rank),  0) +
                   COALESCE(%(kw)s / (%(rrf_k)s + kp.rank), 0) AS rrf_score
            FROM vector_ranked v
            FULL OUTER JOIN trigram_ranked t ON v.id = t.id
            FULL OUTER JOIN dieu_ranked e ON COALESCE(v.id, t.id) = e.id
            FULL OUTER JOIN keyphrase_ranked kp ON COALESCE(v.id, t.id, e.id) = kp.id
        )
        SELECT lc.id, 
                r.rrf_score + CASE WHEN COALESCE(lc.khoan,'') = '' AND COALESCE(lc.diem,'') = '' THEN 0.005 ELSE 0 END AS final_score
        FROM rrf r JOIN law_chunks lc ON lc.id = r.id
        ORDER BY final_score DESC LIMIT %(k)s
    """, {"vec": vec_str, "q": q, "ilike_q": f"%{q}%", "pool": pool,
          "vw": vw, "tw": tw, "dw": dw, "kw": kw, "rrf_k": rrf_k, "k": k})
    return [r[0] for r in cur.fetchall()]

def retrieve_rrf(cur, vec_str, q, k, vw=0.55, tw=0.20, dw=0.25, rrf_k=60):
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
        rrf AS (
            SELECT COALESCE(v.id, t.id, e.id) AS id,
                   COALESCE(%(vw)s / (%(rrf_k)s + v.rank), 0) +
                   COALESCE(%(tw)s / (%(rrf_k)s + t.rank), 0) +
                   COALESCE(%(dw)s / (%(rrf_k)s + e.rank), 0) AS rrf_score
            FROM vector_ranked v
            FULL OUTER JOIN trigram_ranked t ON v.id = t.id
            FULL OUTER JOIN dieu_ranked e ON COALESCE(v.id, t.id) = e.id
        )
        SELECT lc.id, 
                r.rrf_score + CASE WHEN COALESCE(lc.khoan,'') = '' AND COALESCE(lc.diem,'') = '' THEN 0.005 ELSE 0 END AS final_score
        FROM rrf r JOIN law_chunks lc ON lc.id = r.id
        ORDER BY final_score DESC LIMIT %(k)s
    """, {"vec": vec_str, "q": q, "pool": pool, "vw": vw, "tw": tw, "dw": dw, "rrf_k": rrf_k, "k": k})
    return [r[0] for r in cur.fetchall()]

def compute_metrics(ranked: list[int], gold_set: set[int], scores: dict[int, int]) -> dict:
    
    # Multi-gold metrics. scores: chunk_id -> relevance (1 or 2).
    res = {}
    for k in KS:
        topk = ranked[:k]
        hits = [c for c in topk if c in gold_set]
        # Recall@K = |gold ∩ topK| / |gold|
        res[f"recall@{k}"] = len(hits) / len(gold_set) if gold_set else 0.0
        # Hit@K = 1 nếu có ít nhất 1 gold
        res[f"hit@{k}"] = 1.0 if hits else 0.0
        # Precision@K
        res[f"precision@{k}"] = len(hits) / k
        # MRR@K: 1/rank của gold đầu tiên
        mrr = 0.0
        for i, c in enumerate(topk):
            if c in gold_set:
                mrr = 1.0 / (i + 1)
                break
        res[f"mrr@{k}"] = mrr
        # NDCG@K với graded relevance
        dcg = 0.0
        for i, c in enumerate(topk):
            rel = scores.get(c, 0)
            if rel > 0:
                dcg += ((2 ** rel) - 1) / math.log2(i + 2)
        # IDCG = optimal ranking (sort gold by score desc)
        ideal_scores = sorted(scores.values(), reverse=True)[:k]
        idcg = sum(((2 ** r) - 1) / math.log2(i + 2) for i, r in enumerate(ideal_scores) if r > 0)
        res[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-reranker", action = "store_true", help = "Also evaluate vector + BGE-reranker-v2-m3")
    ap.add_argument("--with-keyphrase", action = "store_true", help = "Also evaluate keyphrase_only and hybrid_rrf_keyphrase")
    ap.add_argument("--reranker-pool", type = int, default = 50)
    args = ap.parse_args()

    rows = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("true_gold_chunk_ids")]
    print(f"[re-eval] gold-labeled queries (true_gold non-empty): {len(rows)}")

    print("[re-eval] embedding questions (BGE-M3)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("BAAI/bge-m3", device = device)
    questions = [r["question"] for r in rows]
    t0 = time.time()
    embs = embedder.encode(questions, batch_size = 8, normalize_embeddings = True, convert_to_numpy = True, show_progress_bar = True)
    print(f"[re-eval] embedded in {time.time()-t0:.1f}s")

    reranker = None
    if args.with_reranker:
        print("[re-eval] loading BGE-reranker-v2-m3...")
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)
        if device == "cuda" and reranker is not None: reranker.model.half()  # type: ignore[union-attr]

    results: dict[str, Any] = {"pure_vector": defaultdict(list), "hybrid_rrf": defaultdict(list)}
    per_doc: dict[str, dict] = {"pure_vector": defaultdict(lambda: defaultdict(list)), "hybrid_rrf": defaultdict(lambda: defaultdict(list))}
    if reranker:
        results["vector_rerank_pretrained"] = defaultdict(list)
        per_doc["vector_rerank_pretrained"] = defaultdict(lambda: defaultdict(list))
    if args.with_keyphrase:
        results["keyphrase_only"] = defaultdict(list)
        results["hybrid_rrf_keyphrase"] = defaultdict(list)
        per_doc["keyphrase_only"] = defaultdict(lambda: defaultdict(list))
        per_doc["hybrid_rrf_keyphrase"] = defaultdict(lambda: defaultdict(list))

    latencies: dict[str, list[float]] = defaultdict(list)
    retr_lat: list[float] = []
    rerank_lat: list[float] = []

    t_loop_start = time.time()
    with pg() as conn, conn.cursor() as cur:
        for i, q in enumerate(rows):
            t_query_start = time.perf_counter()
            vec_str = vec_pg(embs[i])
            gold_set = set(q["true_gold_chunk_ids"])
            scores_map = {int(k): int(v) for k, v in (q.get("scores_by_chunk") or {}).items()}
            doc_id = q.get("doc_id")

            # pure_vector
            t0 = time.perf_counter()
            ranked = retrieve_pure_vector(cur, vec_str, q["question"], MAX_K)
            latencies["pure_vector"].append((time.perf_counter() - t0) * 1000)
            m = compute_metrics(ranked, gold_set, scores_map)
            for k_, v_ in m.items(): results["pure_vector"][k_].append(v_)
            for k_, v_ in m.items(): per_doc["pure_vector"][doc_id][k_].append(v_)

            # hybrid_rrf
            t0 = time.perf_counter()
            ranked = retrieve_rrf(cur, vec_str, q["question"], MAX_K)
            latencies["hybrid_rrf"].append((time.perf_counter() - t0) * 1000)
            m = compute_metrics(ranked, gold_set, scores_map)
            for k_, v_ in m.items(): results["hybrid_rrf"][k_].append(v_)
            for k_, v_ in m.items(): per_doc["hybrid_rrf"][doc_id][k_].append(v_)

            # keyphrase_only + hybrid_rrf_keyphrase
            if args.with_keyphrase:
                t0 = time.perf_counter()
                ranked = retrieve_keyphrase(cur, q["question"], MAX_K)
                latencies["keyphrase_only"].append((time.perf_counter() - t0) * 1000)
                m = compute_metrics(ranked, gold_set, scores_map)
                for k_, v_ in m.items(): results["keyphrase_only"][k_].append(v_)
                for k_, v_ in m.items(): per_doc["keyphrase_only"][doc_id][k_].append(v_)

                t0 = time.perf_counter()
                ranked = retrieve_rrf_keyphrase(cur, vec_str, q["question"], MAX_K)
                latencies["hybrid_rrf_keyphrase"].append((time.perf_counter() - t0) * 1000)
                m = compute_metrics(ranked, gold_set, scores_map)
                for k_, v_ in m.items(): results["hybrid_rrf_keyphrase"][k_].append(v_)
                for k_, v_ in m.items(): per_doc["hybrid_rrf_keyphrase"][doc_id][k_].append(v_)

            # vector + reranker
            if reranker:
                t0 = time.perf_counter()
                cur.execute("SELECT id, content FROM law_chunks ORDER BY embedding <=> %s::vector LIMIT %s", (vec_str, args.reranker_pool))
                cands = cur.fetchall()
                retr_lat.append((time.perf_counter() - t0) * 1000)
                cand_ids = [r[0] for r in cands]
                cand_contents = [r[1] or "" for r in cands]
                t0 = time.perf_counter()
                pairs = [(q["question"], c) for c in cand_contents]
                rscores = list(reranker.predict(pairs, batch_size = 32, show_progress_bar = False, activation_fct = None))
                rerank_lat.append((time.perf_counter() - t0) * 1000)
                order = sorted(range(len(rscores)), key = lambda j: -rscores[j])
                ranked = [cand_ids[j] for j in order][:MAX_K]
                m = compute_metrics(ranked, gold_set, scores_map)
                for k_, v_ in m.items(): results["vector_rerank_pretrained"][k_].append(v_)
                for k_, v_ in m.items(): per_doc["vector_rerank_pretrained"][doc_id][k_].append(v_)
                latencies["vector_rerank_pretrained"].append(
                    float(np.mean(retr_lat[-1:] + rerank_lat[-1:])) if (retr_lat and rerank_lat) else 0.0
                )

            if (i + 1) % 10 == 0:
                elapsed_total = time.time() - t_loop_start
                rate = (i + 1) / elapsed_total
                eta_min = (len(rows) - i - 1) / rate / 60
                rrf_lat = latencies["hybrid_rrf"][-1] if latencies["hybrid_rrf"] else 0
                print(f"[re-eval] {i+1}/{len(rows)} | {rate:.2f} q/s | rrf = {rrf_lat:.0f}ms | ETA {eta_min:.1f}min", flush = True)

    # Aggregate
    out: dict[str, Any] = {"test_set_size": len(rows), "ks": KS, "gold_source": "qwen-2.5-7b-instruct", "retrievers": {}, "doc_names": {}}
    for name, metric_lists in results.items():
        summary: dict[str, Any] = {k: round(float(np.mean(v)), 4) for k, v in metric_lists.items()}
        if latencies[name]:
            summary["latency_ms_p50"] = round(float(np.percentile(latencies[name], 50)), 1)
            summary["latency_ms_p95"] = round(float(np.percentile(latencies[name], 95)), 1)
            summary["latency_ms_avg"] = round(float(np.mean(latencies[name])), 1)
        if name == "vector_rerank_pretrained" and retr_lat and rerank_lat:
            summary["retrieve_ms_p50"] = round(float(np.percentile(retr_lat, 50)), 1)
            summary["rerank_ms_p50"] = round(float(np.percentile(rerank_lat, 50)), 1)
        per_doc_out: dict[int, dict[str, Any]] = {}
        for d, mlist in per_doc[name].items():
            if d is None: continue
            row: dict[str, Any] = {"n": len(next(iter(mlist.values())))}
            for k_, v_ in mlist.items():
                row[k_] = round(float(np.mean(v_)), 4)
            per_doc_out[int(d)] = row
        out["retrievers"][name] = {"summary": summary, "per_doc": per_doc_out}

    # Doc names
    with pg() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_code, doc_name FROM documents")
        for r in cur.fetchall():
            out["doc_names"][str(r[0])] = {"code": r[1], "name": r[2]}

    OUT_PATH.write_text(json.dumps(out, ensure_ascii = False, indent = 2), encoding = "utf-8")
    print(f"\n[re-eval] saved: {OUT_PATH}")
    print(json.dumps({n: r["summary"] for n, r in out["retrievers"].items()}, indent = 2, ensure_ascii = False))

if __name__ == "__main__":
    main()
