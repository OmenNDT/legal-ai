"""
Re-label retrieval gold cho test set bằng Qwen 2.5 14B Instruct AWQ (local, RTX 3090).
Mỗi câu hỏi: vector top-20 → listwise prompt (10 chunks/lượt × 2) → Qwen JSON judge → gold_chunk_ids.

Usage:
  python3 llm_judge_label.py [--limit N] [--model MODEL_ID] [--pool 20]
"""
import os
import re
import json
import time
import argparse
from pathlib import Path
import psycopg2
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = Path(__file__).parent
for p in [HERE / "..", HERE / "../..", HERE / "../../..", HERE / "../../../..", Path.cwd()]:
    env_path = (p / ".env").resolve()
    if env_path.is_file():
        load_dotenv(env_path)
        break

TEST_SET = HERE / "test_set_from_split.jsonl"
OUT_PATH = HERE / "gold_labels_qwen.jsonl"

def pg():
    return psycopg2.connect(
        host = os.environ["POSTGRES_HOST"].strip(),
        port = int(os.environ["POSTGRES_PORT"].strip()),
        dbname = os.environ["POSTGRES_DB"].strip(),
        user = os.environ["POSTGRES_USER"].strip(),
        password = os.environ["POSTGRES_PASSWORD"].strip()
    )

def vec_pg(v): return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

PROMPT_TEMPLATE = """Bạn là chuyên gia pháp luật Việt Nam, đánh giá mức độ liên quan giữa câu hỏi pháp lý và các điều khoản luật.

CÂU HỎI: {question}

CÁC ĐIỀU KHOẢN (đánh số):
{chunks}

Với MỖI điều khoản, đánh giá mức độ liên quan:
- 2 = TRỰC TIẾP trả lời / là căn cứ pháp lý chính cho câu hỏi
- 1 = LIÊN QUAN (cùng chủ đề, có thể là bối cảnh, định nghĩa, hoặc trả lời một phần)
- 0 = KHÔNG LIÊN QUAN

QUAN TRỌNG: Hãy CỞI MỞ trong đánh giá. Nếu điều khoản đề cập đến cùng chủ đề / đối tượng / hành vi với câu hỏi, đánh giá ÍT NHẤT 1. Chỉ đánh 0 khi điều khoản hoàn toàn về chủ đề khác.

Trả lời CHỈ bằng JSON (không thêm chữ nào khác):
{{"scores": {{"1": 2, "2": 1, "3": 0, ...}}, "best_id": id có điểm cao nhất, "reason": "lý do ngắn 1 câu"}}"""

JSON_RE = re.compile(r"\{[\s\S]*\"scores\"[\s\S]*\}")

def build_chunk_block(chunks: list[tuple[int, str]], offset: int = 0, max_chars: int = 600) -> str:
    lines = []
    for i, (_, content) in enumerate(chunks, start = offset + 1):
        snippet = (content or "").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        lines.append(f"[{i}] {snippet}")
    return "\n\n".join(lines)

def parse_judge_output(text: str) -> dict | None:
    # Try parsing the whole text first as JSON
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type = int, default = None)
    ap.add_argument("--model", default = "Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pool", type = int, default = 20)
    ap.add_argument("--batch-per-prompt", type = int, default = 10)
    ap.add_argument("--resume", action = "store_true", help = "Skip qa_ids already in output file")
    args = ap.parse_args()

    rows = [json.loads(l) for l in TEST_SET.read_text(encoding = "utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[judge] test set: {len(rows)} queries, model = {args.model}")

    done_qa_ids: set[int] = set()
    if args.resume and OUT_PATH.exists():
        for l in OUT_PATH.read_text(encoding = "utf-8").splitlines():
            try:
                done_qa_ids.add(json.loads(l)["qa_id"])
            except Exception:
                pass
        print(f"[judge] resume: skipping {len(done_qa_ids)} already-judged qa_ids")
        rows = [r for r in rows if r["qa_id"] not in done_qa_ids]
        print(f"[judge] remaining: {len(rows)}")

    print("[judge] loading BGE-M3 embedder...")
    embedder = SentenceTransformer("BAAI/bge-m3", device = "cuda" if torch.cuda.is_available() else "cpu")

    print(f"[judge] loading {args.model}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map = "cuda", torch_dtype = "auto")
    model.eval()
    print(f"[judge] LLM loaded in {time.time() - t0:.1f}s")

    out_f = OUT_PATH.open("a", encoding = "utf-8")
    t_start = time.time()
    for i, q in enumerate(rows):
        question = q["question"]
        q_vec = embedder.encode(question, normalize_embeddings = True, convert_to_numpy = True)
        vec_str = vec_pg(q_vec)

        with pg() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id, content FROM law_chunks
                ORDER BY embedding <=> %s::vector LIMIT %s
            """, (vec_str, args.pool))
            cands = cur.fetchall()
        cand_ids = [r[0] for r in cands]
        cand_contents = [r[1] or "" for r in cands]

        # Split into windows of batch-per-prompt (10) and judge each window
        scores_by_chunk: dict[int, int] = {}
        reasons: list[str] = []
        raw_outputs: list[str] = []
        for start in range(0, len(cand_ids), args.batch_per_prompt):
            window_ids = cand_ids[start : start + args.batch_per_prompt]
            window_contents = cand_contents[start : start + args.batch_per_prompt]
            window_pairs = list(zip(window_ids, window_contents))
            chunk_block = build_chunk_block(window_pairs, offset = start)
            prompt = PROMPT_TEMPLATE.format(question = question, chunks = chunk_block)
            messages = [{"role": "user", "content": prompt}]
            chat = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
            inputs = tokenizer(chat, return_tensors = "pt", truncation = True, max_length = 8000).to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens = 300,
                    do_sample = False,
                    temperature = 1.0,
                    pad_token_id = tokenizer.eos_token_id
                )
            output_ids = gen[0][inputs.input_ids.shape[1]:]
            text = tokenizer.decode(output_ids, skip_special_tokens = True)
            raw_outputs.append(text)
            parsed = parse_judge_output(text)
            if parsed:
                scores_dict = parsed.get("scores") or {}
                for sid, sval in scores_dict.items():
                    try:
                        idx = int(sid) - 1
                        score = int(sval)
                        if 0 <= idx < len(cand_ids):
                            scores_by_chunk[cand_ids[idx]] = max(scores_by_chunk.get(cand_ids[idx], 0), score)
                    except Exception:
                        pass
                reasons.append(parsed.get("reason", "") or "")

        all_relevant = sorted([cid for cid, s in scores_by_chunk.items() if s >= 1])
        highly_relevant = sorted([cid for cid, s in scores_by_chunk.items() if s >= 2])
        best_overall = max(scores_by_chunk.items(), key = lambda kv: kv[1])[0] if scores_by_chunk else None

        record = {
            "qa_id": q["qa_id"],
            "question": question,
            "answer": q.get("answer", ""),
            "doc_id": q["doc_id"],
            "doc_code": q.get("doc_code", ""),
            "weak_gold_chunk_id": q["weak_gold_chunk_id"],
            "weak_gold_similarity": q["weak_gold_similarity"],
            "true_gold_chunk_ids": all_relevant,
            "highly_relevant_chunk_ids": highly_relevant,
            "scores_by_chunk": scores_by_chunk,
            "best_chunk_id": best_overall,
            "candidate_pool": cand_ids,
            "judge_reasons": reasons,
            "raw_outputs": raw_outputs
        }
        out_f.write(json.dumps(record, ensure_ascii = False) + "\n")
        out_f.flush()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remain = (len(rows) - i - 1) / rate
            print(f"  [judge] {i+1}/{len(rows)} | {rate:.2f} q/s | ETA {remain/60:.1f}min")

    out_f.close()
    print(f"\n[judge] saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
