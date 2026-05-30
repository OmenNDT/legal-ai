import json
import pickle
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
SRC = HERE / "test_set.jsonl"
DST = HERE / "query_embeddings.pkl"

def main():
    rows = [json.loads(l) for l in SRC.read_text(encoding = "utf-8").splitlines() if l.strip()]
    questions = [r["question"] for r in rows]
    qa_ids = [r["qa_id"] for r in rows]
    print(f"[embed] {len(questions)} questions to embed")

    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"[embed] model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    vecs = model.encode(
        questions,
        batch_size = 8,
        normalize_embeddings = True,
        show_progress_bar = True,
        convert_to_numpy = True
    )
    print(f"[embed] encoded in {time.time() - t0:.1f}s, shape = {vecs.shape}")

    with DST.open("wb") as f:
        pickle.dump({"qa_ids": qa_ids, "vectors": vecs}, f)
    print(f"[embed] saved: {DST}")

if __name__ == "__main__":
    main()
