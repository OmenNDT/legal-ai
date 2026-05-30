import json
import argparse
from pathlib import Path

HERE = Path(__file__).parent

RETRIEVER_LABELS = {
    "pure_vector": "Pure Vector",
    "hybrid_rrf": "Hybrid RRF",
    "vector_rerank_pretrained": "Vector + Reranker",
}

METRIC_GROUPS = {
    "Recall@K": "recall",
    "Precision@K": "precision",
    "MRR@K": "mrr",
    "NDCG@K": "ndcg"
}

def _col(val: float | None, width: int = 9) -> str:
    if val is None:
        return " " * width
    return f"{val:.4f}".rjust(width)

def _fmt(val: float | None) -> str:
    return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

def _lat(val: float | None) -> str:
    return f"{val:.1f}".rjust(9) if isinstance(val, (int, float)) else " " * 9

def _sep(widths):
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

def _row(cells, widths):
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(f" {str(cell).ljust(w)} ")
    return "|" + "|".join(parts) + "|"

def print_table(title, headers, rows):
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = _sep(widths)
    print(f"\n  {title}")
    print(sep)
    print(_row(headers, widths))
    print(sep)
    for row in rows:
        print(_row(row, widths))
    print(sep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default = None, help = "Path to results_baseline.json")
    ap.add_argument("--ks", nargs = "+", type = int, default = [1, 5, 10, 20], help = "K values to display")
    ap.add_argument("--out", default = None, help = "Path to save summary JSON")
    args = ap.parse_args()

    path = Path(args.file) if args.file else HERE.parent / "model" / "eval_results" / "results_baseline.json"
    if not path.exists():
        path = HERE / "results_baseline.json"
    if not path.exists():
        print(f"[!] Không tìm thấy file kết quả. Chạy re_eval_with_true_gold.py trước.")

    data = json.loads(path.read_text(encoding = "utf-8"))
    retrievers = data["retrievers"]
    ks = [k for k in args.ks if k in data.get("ks", args.ks)]
    names = list(retrievers.keys())
    labels = [RETRIEVER_LABELS.get(n, n) for n in names]

    # Per-metric bảng: rows = K values, cols = retrievers
    for metric_label, metric_key in METRIC_GROUPS.items():
        headers = [f"K"] + labels
        rows = []
        for k in ks:
            key = f"{metric_key}@{k}"
            row = [str(k)]
            for name in names:
                val = retrievers[name]["summary"].get(key)
                row.append(_col(val))
            rows.append(row)
        print_table(metric_label, headers, rows)

    # Latency
    print("\nLatency (ms)")
    lat_headers = ["Metric"] + labels
    lat_rows = []
    for lkey, llabel in [("latency_ms_p50", "p50"), ("latency_ms_p95", "p95"), ("latency_ms_avg", "avg")]:
        row = [llabel]
        for name in names:
            val = retrievers[name]["summary"].get(lkey)
            row.append(_lat(val))
        lat_rows.append(row)
    # reranker breakdown
    for name in names:
        s = retrievers[name]["summary"]
        if "retrieve_ms_p50" in s:
            idx = names.index(name)
            blanks_before = ["" for _ in range(idx)]
            blanks_after = ["" for _ in range(len(names) - idx - 1)]
            lat_rows.append(
                [f"retrieve p50 ({(RETRIEVER_LABELS.get(name) or name)[:20]})"] +
                blanks_before + [_lat(s.get("retrieve_ms_p50"))] + blanks_after
            )
            lat_rows.append(
                [f"rerank p50 ({(RETRIEVER_LABELS.get(name) or name)[:20]})"] +
                blanks_before + [_lat(s.get("rerank_ms_p50"))] + blanks_after
            )
    print_table("Latency (ms)", lat_headers, lat_rows)
    best_k = min(ks, key=lambda k: (abs(k - 10), k > 10))
    print(f"\nTỔNG KẾT @ K = {best_k}")
    print("  " + "-" * 60)
    for name, label in zip(names, labels):
        s = retrievers[name]["summary"]
        print(f"{label}")
        print(
            f"Recall@{best_k} = {_fmt(s.get(f'recall@{best_k}'))}  "
            f"MRR@{best_k} = {_fmt(s.get(f'mrr@{best_k}'))}  "
            f"Precision@{best_k} = {_fmt(s.get(f'precision@{best_k}'))}  "
            f"NDCG@{best_k} = {_fmt(s.get(f'ndcg@{best_k}'))}")
    print("  " + "-" * 60)
    print()
    print(f"  Tại sao K = {best_k} là ngưỡng chính?")
    print(f"  - Pipeline RAG hiện tại đưa top-{best_k} chunk vào BARTpho để sinh câu trả lời.")
    print(f"  - K < {best_k}: recall thấp, bỏ sót chunk quan trọng.")
    print(f"  - K > {best_k}: context quá dài, model bị nhiễu, chất lượng câu trả lời giảm.")
    print(f"  - K = {best_k} là điểm cân bằng giữa recall và độ chính xác cho pipeline này.")
    print()

    # Ghi JSON summary
    out_path = Path(args.out) if args.out else path.parent / "eval_summary.json"
    summary: dict = {"best_k": best_k, "retrievers": {}}
    for name, label in zip(names, labels):
        s = retrievers[name]["summary"]
        summary["retrievers"][label] = {
            f"recall@{k}": s.get(f"recall@{k}") for k in ks
        } | {
            f"mrr@{k}": s.get(f"mrr@{k}") for k in ks
        } | {
            f"precision@{k}": s.get(f"precision@{k}") for k in ks
        } | {
            f"ndcg@{k}": s.get(f"ndcg@{k}") for k in ks
        } | {
            "latency_ms_p50": s.get("latency_ms_p50"),
            "latency_ms_p95": s.get("latency_ms_p95"),
            "latency_ms_avg": s.get("latency_ms_avg")
        }
    out_path.write_text(json.dumps(summary, ensure_ascii = False, indent = 2), encoding = "utf-8")
    print(f"Đã lưu: {out_path}")
