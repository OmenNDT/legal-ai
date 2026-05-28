"""Slim Flask backend for string-matching service only.

Chỉ phục vụ:
  - GET  /                         -> redirect tới /legal-ai/ (do APP_PREFIX)
  - GET  /legal-ai/, /legal-ai/<f> -> serve frontend dist (SPA fallback)
  - POST /api/string-matching/naive
  - POST /api/string-matching/kmp
  - POST /api/string-matching/boyer-moore
  - POST /api/string-matching/export
  - GET  /health

Bỏ qua toàn bộ RAG / KG / Search / Auth của app.py gốc — những module đó
không có trong folder này.
"""
from __future__ import annotations

import os
import re as _re
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── relative import từ chính package backend (không phải `backend.string_matching.*`) ──
from .naive.naive_matcher import NaiveMatcher
from .kmp.kmp_matcher import KMPMatcher
from .boyer_moore.boyer_moore_matcher import BoyerMooreMatcher
from .auth_slim import auth_bp, init_pool as init_auth_pool

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIST) if FRONTEND_DIST.exists() else None,
    static_url_path="",
)
CORS(app)

app.register_blueprint(auth_bp)
try:
    init_auth_pool()
    print("Auth DB pool initialized (legal_ai).")
except Exception as exc:
    print(f"WARN: Auth DB pool init failed: {exc}")


def _run_matcher(matcher_cls, algo_key):
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "") or ""
    pattern = payload.get("pattern", "") or ""
    case_sensitive = bool(payload.get("case_sensitive", False))
    trace = bool(payload.get("trace", True))

    if not pattern:
        return jsonify({"error": "Pattern must not be empty."}), 400

    matcher = matcher_cls(case_sensitive=case_sensitive)
    t0 = time.perf_counter()
    res = matcher.search(text, pattern, trace=trace)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    d = res.to_dict()
    d["algorithm"] = algo_key
    d["elapsed_ms"] = elapsed_ms
    return jsonify({"ok": True, "result": d})


@app.post("/api/string-matching/naive")
def sm_naive():
    return _run_matcher(NaiveMatcher, "naive")


@app.post("/api/string-matching/kmp")
def sm_kmp():
    return _run_matcher(KMPMatcher, "kmp")


@app.post("/api/string-matching/boyer-moore")
def sm_bm():
    return _run_matcher(BoyerMooreMatcher, "boyer_moore")


@app.post("/api/string-matching/export")
def sm_export():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "") or ""
    pattern = payload.get("pattern", "") or ""
    case_sensitive = bool(payload.get("case_sensitive", False))
    positions = payload.get("positions") or []
    occurrences = payload.get("occurrences")
    if occurrences is None:
        occurrences = len(positions)
    complexities = payload.get("complexities") or {}
    comparisons = payload.get("comparisons") or {}

    short_names = [("naive", "Naïve"), ("kmp", "KMP"), ("boyer_moore", "BM")]
    selected_keys = [k for k, _ in short_names if k in complexities or k in comparisons]

    lines = []
    selected_label = ", ".join(label for k, label in short_names if k in selected_keys) or "—"
    lines.append(f"Tìm kiếm văn bản: Quan sát {selected_label}, tìm pattern trong text.")
    lines.append("")
    lines.append("Text (chuỗi nguồn):")
    lines.append(text)
    lines.append("")
    lines.append("Pattern (chuỗi cần tìm):")
    lines.append(pattern)
    lines.append("")
    lines.append(f"Case-sensitive (Y/N): {'Y' if case_sensitive else 'N'}")
    lines.append("")
    lines.append("Kết quả:")
    lines.append(f"- Vị trí tìm thấy: {', '.join(str(p) for p in positions) if positions else '—'}")
    lines.append(f"- Số lần xuất hiện: {occurrences}")
    lines.append("- Độ phức tạp (worst case):")
    for k, label in short_names:
        if k in selected_keys:
            lines.append(f"\t+ {label}: {complexities.get(k, '—')}")
    lines.append("- Số phép so sánh:")
    for k, label in short_names:
        if k in selected_keys:
            lines.append(f"\t+ {label}: {comparisons.get(k, '—')}")
    content = "\n".join(lines) + "\n"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_pattern = _re.sub(r"[^a-zA-Z0-9_-]+", "_", pattern.strip())[:30] or "pattern"
    filename = f"result_{ts}_{safe_pattern}.txt"
    out_path = RESULT_DIR / filename
    out_path.write_text(content, encoding="utf-8")

    return jsonify({"ok": True, "filename": filename, "path": str(out_path), "content": content})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "string-matching"})


# ── Frontend SPA: phải đặt SAU API routes để không trùng /api/* ──
@app.get("/")
def _serve_index():
    if not FRONTEND_DIST.exists():
        return jsonify({"error": "Frontend not built."}), 503
    return send_from_directory(str(FRONTEND_DIST), "index.html")


@app.get("/<path:filename>")
def _serve_static(filename: str):
    if not FRONTEND_DIST.exists():
        return jsonify({"error": "Frontend not built."}), 503
    target = FRONTEND_DIST / filename
    if target.is_file():
        return send_from_directory(str(FRONTEND_DIST), filename)
    return send_from_directory(str(FRONTEND_DIST), "index.html")


# ── APP_PREFIX wrapper (frontend build với base='/legal-ai/') ──
APP_PREFIX = os.environ.get("APP_PREFIX", "/legal-ai")
if APP_PREFIX and APP_PREFIX != "/":
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.wrappers import Response as WResponse

    inner_app = app.wsgi_app

    def _redirect_root(environ, start_response):
        return WResponse("", status=302, headers={"Location": APP_PREFIX + "/"})(environ, start_response)

    app.wsgi_app = DispatcherMiddleware(_redirect_root, {APP_PREFIX: inner_app})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9010, debug=False)
