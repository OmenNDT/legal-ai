import os
import json
import pickle
import time
from pathlib import Path
from collections import defaultdict
from typing import Any
import math
import numpy as np
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

HERE = Path(__file__).parent
for p in [HERE / "..", HERE / "../..", HERE / "../../..", HERE / "../../../..", Path.cwd()]:
    env_path = (p / ".env").resolve()
    if env_path.is_file():
        load_dotenv(env_path)
        break

TEST_SET = HERE / "test_set.jsonl"
EMB_CACHE = HERE / "query_embeddings.pkl"
POOL = 50
KS = [1, 3, 5, 10, 20]
MAX_K = max(KS)

def pg_connect():
    return psycopg2.connect(
        host = os.environ["POSTGRES_HOST"].strip(),
        port = int(os.environ["POSTGRES_PORT"].strip()),
        dbname = os.environ["POSTGRES_DB"].strip(),
        user = os.environ["POSTGRES_USER"].strip(),
        password = os.environ["POSTGRES_PASSWORD"].strip()
    )

def vec_pg(v): return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

def answer_overlap(content: str, answer: str, min_words: int = 4) -> bool:
    if not content or not answer: return False
    a = " ".join(answer.lower().split())
    c = " ".join(content.lower().split())
    w = a.split()
    if len(w) < min_words: return a in c
    for i in range(len(w) - min_words + 1):
        if " ".join(w[i:i + min_words]) in c: return True
    return False

def main():
    rows = [json.loads(l) for l in TEST_SET.read_text(encoding = "utf-8").splitlines() if l.strip()]
    with EMB_CACHE.open("rb") as f:
        cache = pickle.load(f)
    qa2vec = dict(zip(cache["qa_ids"], cache["vectors"]))
    embs = np.stack([qa2vec[r["qa_id"]] for r in rows])
    print(f"[reranker] {len(rows)} queries, embeddings {embs.shape}")

    print("[reranker] loading BAAI/bge-reranker-v2-m3 (GPU)...")
    t0 = time.time()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device = device, max_length = 512)
    if device == "cuda" and reranker.model is not None:
        reranker.model.half() # fp16
    print(f"[reranker] loaded in {time.time()-t0:.1f}s on {device}")

    per_doc = defaultdict(lambda: {"n": 0, **{f"hit@{k}": 0 for k in KS}})
    summary: dict[str, Any] = {f"recall@{k}": 0 for k in KS}
    summary.update({f"mrr@{k}": 0.0 for k in KS})
    summary.update({f"ndcg@{k}": 0.0 for k in KS})
    summary["found_in_top"] = 0
    summary["overlap_in_top5"] = 0
    summary["overlap_in_top10"] = 0
    rank_sum = 0.0; rank_count = 0
    retrieve_lat = []; rerank_lat = []

    with pg_connect() as conn, conn.cursor() as cur:
        for i, q in enumerate(rows):
            vec_str = vec_pg(embs[i])
            t0 = time.perf_counter()
            cur.execute("""
                SELECT id, content FROM law_chunks
                ORDER BY embedding <=> %s::vector LIMIT %s
            """, (vec_str, POOL))
            cands = cur.fetchall()
            retrieve_lat.append((time.perf_counter() - t0) * 1000)
            cand_ids = [r[0] for r in cands]
            cand_contents = [r[1] or "" for r in cands]

            t0 = time.perf_counter()
            pairs = [(q["question"], c) for c in cand_contents]
            scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False, activation_fct=None)
            scores = list(scores) if hasattr(scores, "__iter__") else [scores]
            rerank_lat.append((time.perf_counter() - t0) * 1000)

            order = sorted(range(len(scores)), key=lambda j: -scores[j])
            ranked = [cand_ids[j] for j in order][:MAX_K]
            ranked_contents = [cand_contents[j] for j in order][:10]

            gold = q["gold_chunk_id"]
            if gold in ranked:
                r = ranked.index(gold) + 1
                rank_sum += r; rank_count += 1
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

            if any(answer_overlap(c, q["answer"]) for c in ranked_contents[:5]):
                summary["overlap_in_top5"] += 1
            if any(answer_overlap(c, q["answer"]) for c in ranked_contents[:10]):
                summary["overlap_in_top10"] += 1

            if (i + 1) % 25 == 0:
                print(f"[reranker] {i+1}/{len(rows)}")

    n = len(rows)
    for k in KS:
        summary[f"recall@{k}"] = round(summary[f"recall@{k}"] / n, 4)
        summary[f"mrr@{k}"] = round(summary[f"mrr@{k}"] / n, 4)
        summary[f"ndcg@{k}"] = round(summary[f"ndcg@{k}"] / n, 4)
    summary["mean_rank_of_gold"] = round(rank_sum / rank_count, 2) if rank_count else None
    summary["found_in_top"] = round(summary["found_in_top"] / n, 4)
    summary["overlap_in_top5"] = round(summary["overlap_in_top5"] / n, 4)
    summary["overlap_in_top10"] = round(summary["overlap_in_top10"] / n, 4)
    summary["retrieve_ms_p50"] = round(float(np.percentile(retrieve_lat, 50)), 1)
    summary["rerank_ms_p50"] = round(float(np.percentile(rerank_lat, 50)), 1)
    summary["rerank_ms_p95"] = round(float(np.percentile(rerank_lat, 95)), 1)
    summary["latency_ms_p50"] = round(summary["retrieve_ms_p50"] + summary["rerank_ms_p50"], 1)

    per_doc_out: dict[int, dict[str, Any]] = {}
    for d, st in per_doc.items():
        row: dict[str, Any] = {"n": st["n"]}
        for k in KS:
            row[f"recall@{k}"] = round(st[f"hit@{k}"] / st["n"], 4)
        per_doc_out[int(d)] = row

    # Merge into results_baseline.json
    res_path = HERE / "results_baseline.json"
    if res_path.exists():
        out = json.loads(res_path.read_text(encoding="utf-8"))
    else:
        out = {"test_set_size": n, "ks": KS, "retrievers": {}, "doc_names": {}}
    out["retrievers"]["vector_rerank_pretrained"] = {"summary": summary, "per_doc": per_doc_out}
    res_path.write_text(json.dumps(out, ensure_ascii = False, indent = 2), encoding = "utf-8")
    print(f"\n[reranker] merged into {res_path}")
    print(json.dumps(summary, indent = 2, ensure_ascii = False))

if __name__ == "__main__":
    main()
