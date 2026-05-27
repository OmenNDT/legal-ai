import os
import torch
from typing import Any, Optional, Tuple
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Service nhẹ chạy trên worker1 (GPU). Master forward request sang đây để tận dụng RTX 3090.

# Model path - mặc định trỏ checkpoint fine-tune v3, override qua env
MODEL_PATH = os.environ.get(
    "BART_GPU_MODEL",
    "/home/sontn/text_sumarisation/outputs/bart-cuad-v3/final"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = int(os.environ.get("BART_GPU_PORT", "9021"))

app = Flask(__name__)

# Cache singleton để model load 1 lần
_tokenizer: Optional[Any] = None
_model: Optional[Any] = None

def _load() -> Tuple[Any, Any]:
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        print(f"[gpu_worker] Loading {MODEL_PATH} on {DEVICE} ...", flush = True)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        m = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        m.to(DEVICE)
        m.eval()
        _model = m
        print("[gpu_worker] Model loaded.", flush = True)
    return _tokenizer, _model

@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "model": MODEL_PATH, "loaded": _model is not None})

# Tóm tắt 1 chunk text - input đã được master cắt vừa max_input token
@app.post("/summarize_chunk")
def summarize_chunk():
    body = request.get_json(silent = True) or {}
    text = body.get("text") or ""
    if not text:
        return jsonify({"error": "Thiếu text"}), 400
    max_input = int(body.get("max_input", 1024))
    max_output = int(body.get("max_output", 256))
    min_output = int(body.get("min_output", 80))
    num_beams = int(body.get("num_beams", 4))
    tokenizer, model = _load()
    inputs = tokenizer(
        text,
        return_tensors = "pt",
        max_length = max_input,
        truncation = True,
    ).to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_length = max_output,
            min_length = min_output,
            num_beams = num_beams,
            length_penalty = 2.0,
            early_stopping = True,
            no_repeat_ngram_size = 3
        )
    summary = tokenizer.decode(ids[0], skip_special_tokens = True).strip()
    return jsonify({"summary": summary})

# Đếm số token (master cần để biết khi nào cần hierarchical lần 2)
@app.post("/count_tokens")
def count_tokens():
    body = request.get_json(silent = True) or {}
    text = body.get("text") or ""
    tokenizer, _ = _load()
    n = len(tokenizer.encode(text))
    return jsonify({"num_tokens": n})

if __name__ == "__main__":
    # Pre-load để health check đầu tiên không bị chờ
    _load()
    app.run(host = "0.0.0.0", port = PORT, debug = False, threaded = False)
