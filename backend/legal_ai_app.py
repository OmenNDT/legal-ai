import os
import sys
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import threading
from psycopg2 import pool as _pg_pool
from contextlib import contextmanager

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
TEXT_SUM_ROOT = ROOT / "backend" / "text_sumarisation"
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.get("/")
def _serve_index():
    if not FRONTEND_DIST.exists():
        return jsonify({"error": "Frontend not built. Run: cd frontend && npm run build"}), 503
    return send_from_directory(str(FRONTEND_DIST), "index.html")

@app.get("/<path:filename>")
def _serve_static(filename: str):
    if not FRONTEND_DIST.exists():
        return jsonify({"error": "Frontend not built."}), 503
    target = FRONTEND_DIST / filename
    if target.is_file():
        return send_from_directory(str(FRONTEND_DIST), filename)
    return send_from_directory(str(FRONTEND_DIST), "index.html")

def _run_matcher(matcher_cls, algo_key):
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "") or ""
    pattern = payload.get("pattern", "") or ""
    case_sensitive = bool(payload.get("case_sensitive", False))
    trace = bool(payload.get("trace", True))
    if not pattern:
        return jsonify({"error": "Pattern must not be empty."}), 400
    matcher = matcher_cls(case_sensitive = case_sensitive)
    t0 = time.perf_counter()
    res = matcher.search(text, pattern, trace = trace)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    d = res.to_dict()
    d["algorithm"] = algo_key
    d["elapsed_ms"] = elapsed_ms
    return jsonify({"ok": True, "result": d})

@app.post("/api/string-matching/naive")
def string_matching_naive():
    from backend.string_matching.naive.naive_matcher import NaiveMatcher
    return _run_matcher(NaiveMatcher, "naive")

@app.post("/api/string-matching/kmp")
def string_matching_kmp():
    from backend.string_matching.kmp.kmp_matcher import KMPMatcher
    return _run_matcher(KMPMatcher, "kmp")

@app.post("/api/string-matching/boyer-moore")
def string_matching_boyer_moore():
    from backend.string_matching.boyer_moore.boyer_moore_matcher import BoyerMooreMatcher
    return _run_matcher(BoyerMooreMatcher, "boyer_moore")

@app.post("/api/string-matching/export")
def string_matching_export():
    import re as _re
    from datetime import datetime
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
    out_dir = ROOT / "backend" / "string_matching" / "result"
    out_dir.mkdir(parents = True, exist_ok = True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_pattern = _re.sub(r"[^a-zA-Z0-9_-]+", "_", pattern.strip())[:30] or "pattern"
    filename = f"result_{ts}_{safe_pattern}.txt"
    out_path = out_dir / filename
    out_path.write_text(content, encoding = "utf-8")
    return jsonify({"ok": True, "filename": filename, "path": str(out_path), "content": content})

def _register_summarisation_blueprints():
    sum_root = str(TEXT_SUM_ROOT)
    original_path = sys.path[:]
    sys.path = [sum_root] + [p for p in sys.path if p != sum_root]
    for k in [k for k in sys.modules if k == "backend" or k.startswith("backend.")]:
        del sys.modules[k]
    try:
        import backend.app.routes.summarize as ts_summarize
        import backend.app.routes.extract as ts_extract
        import backend.app.routes.documents as ts_documents
        import backend.app.routes.eval as ts_eval
        import backend.app.auth as ts_auth
        from backend.app.state import AppState
        from backend.config.settings import get_settings
        from backend.utils.logger import Logger

        settings = get_settings()
        Logger.setup(settings.LOG_DIR, name = "api")
        AppState.instance()

        app.register_blueprint(ts_summarize.bp)
        app.register_blueprint(ts_extract.bp)
        app.register_blueprint(ts_documents.bp)
        app.register_blueprint(ts_eval.bp)
        app.register_blueprint(ts_auth.bp)
        print("[legal-ai] Text summarisation blueprints registered.")
    except Exception as exc:
        print(f"[legal-ai] WARNING: Could not load text summarisation module: {exc}")
        import traceback; traceback.print_exc()
    finally:
        project_root = str(ROOT)
        sys.path = original_path
        if sum_root not in sys.path:
            sys.path.append(sum_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

_register_summarisation_blueprints()

_pg_pool_instance = _pg_pool.ThreadedConnectionPool(
    minconn = 1, maxconn = 10,
    host = os.environ["POSTGRES_HOST"],
    port = int(os.environ.get("POSTGRES_PORT", 5432)),
    dbname = os.environ["POSTGRES_DB"],
    user = os.environ["POSTGRES_USER"],
    password = os.environ["POSTGRES_PASSWORD"]
)

@contextmanager
def _pg_connect():
    conn = _pg_pool_instance.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pg_pool_instance.putconn(conn)

import atexit
@atexit.register
def _close_pg_pool():
    try:
        if _pg_pool_instance and not _pg_pool_instance.closed:
            _pg_pool_instance.closeall()
            print("[legal-ai] PostgreSQL pool closed.")
    except Exception as e:
        print(f"[legal-ai] Error closing pool: {e}")

def _doc_to_dict(row):
    return {
        "id": row[0],
        "doc_code": row[1],
        "doc_name": row[2],
        "doc_type": row[3],
        "issue_year": row[4],
        "chunk_count": row[5]
    }

@app.get("/api/laws")
def api_list_documents():
    q = (request.args.get("q") or "").strip()
    try:
        with _pg_connect() as conn, conn.cursor() as cur:
            if q:
                cur.execute("""
                    SELECT d.id, d.doc_code, d.doc_name, d.doc_type, d.issue_year,
                           COUNT(c.id) AS chunk_count
                    FROM documents d
                    LEFT JOIN law_chunks c ON c.document_id = d.id
                    WHERE unaccent(lower(d.doc_name)) ILIKE unaccent(lower(%s))
                       OR unaccent(lower(d.doc_type)) ILIKE unaccent(lower(%s))
                       OR CAST(d.issue_year AS TEXT) LIKE %s
                    GROUP BY d.id
                    ORDER BY d.issue_year DESC NULLS LAST, d.doc_name
                """, (f"%{q}%", f"%{q}%", f"%{q}%"))
            else:
                cur.execute("""
                    SELECT d.id, d.doc_code, d.doc_name, d.doc_type, d.issue_year,
                           COUNT(c.id) AS chunk_count
                    FROM documents d
                    LEFT JOIN law_chunks c ON c.document_id = d.id
                    GROUP BY d.id
                    ORDER BY d.issue_year DESC NULLS LAST, d.doc_name
                """)
            rows = cur.fetchall()
        return jsonify({"ok": True, "documents": [_doc_to_dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/laws/<int:doc_id>")
def api_get_document(doc_id: int):
    try:
        with _pg_connect() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id, doc_code, doc_name, doc_type, issue_year
                FROM documents WHERE id = %s
            """, (doc_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"ok": False, "error": "Not found"}), 404
            meta = {
                "id": row[0], "doc_code": row[1], "doc_name": row[2],
                "doc_type": row[3], "issue_year": row[4]
            }
            cur.execute("""
                SELECT dieu, COUNT(*) AS cnt, MIN(id) AS first_id
                FROM law_chunks
                WHERE document_id = %s AND dieu IS NOT NULL AND dieu <> ''
                GROUP BY dieu
                ORDER BY first_id
            """, (doc_id,))
            outline = [
                {"dieu_key": r[0], "title": r[0], "chunk_count": r[1]}
                for r in cur.fetchall()
            ]
        meta["outline"] = outline
        meta["chunk_count"] = sum(o["chunk_count"] for o in outline)
        return jsonify({"ok": True, "document": meta})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/laws/<int:doc_id>/dieu")
def api_get_dieu(doc_id: int):
    dieu_key = (request.args.get("key") or "").strip()
    if not dieu_key:
        return jsonify({"ok": False, "error": "Missing key"}), 400
    try:
        with _pg_connect() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT khoan, diem, content
                FROM law_chunks
                WHERE document_id = %s AND dieu = %s
                ORDER BY id
            """, (doc_id, dieu_key))
            rows = cur.fetchall()

        parts = [dieu_key]
        title_seen = False
        for khoan, diem, content in rows:
            c = (content or "").strip()
            if not c:
                continue
            if not khoan and not diem:
                # Tiêu đề / đoạn mở đầu của Điều
                if not title_seen and c.startswith(dieu_key):
                    # Chunk đã chứa tiêu đề — thay vì lặp, dùng phần sau
                    rest = c[len(dieu_key):].strip()
                    if rest:
                        parts.append(rest)
                else:
                    parts.append(c)
                title_seen = True
                continue
            prefix_bits = []
            if khoan: prefix_bits.append(khoan)
            if diem: prefix_bits.append(diem)
            prefix = " ".join(prefix_bits)
            parts.append(f"{prefix}. {c}" if prefix else c)

        content = "\n\n".join(parts) if len(parts) > 1 else "(Chưa có nội dung)"
        return jsonify({"ok": True, "dieu_key": dieu_key, "content": content, "chunk_count": len(rows)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

_chat_state = {
    "pipeline": None,
    "status": "idle",  # idle | loading | ready | error
    "error": None,
    "mode": "extractive"
}
_chat_lock = threading.Lock()

def _init_chat_pipeline_blocking():
    try:
        # _register_summarisation_blueprints rewrites sys.modules['backend'] to
        # point at backend/text_sumarisation/backend — purge those entries so
        # the import below resolves the project root backend package.
        project_root = str(ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        for k in list(sys.modules):
            if k == "backend" or k.startswith("backend."):
                mod = sys.modules.get(k)
                mod_file = getattr(mod, "__file__", "") or ""
                if "text_sumarisation" in mod_file:
                    del sys.modules[k]

        from backend.chatbot.data_pipeline.db_loader import DbConfig
        from backend.chatbot.data_pipeline.embedder import Embedder
        from backend.chatbot.inference.retriever import Retriever
        from backend.chatbot.inference.prompt_builder import PromptBuilder
        from backend.chatbot.inference.generator import Generator, GeneratorConfig
        from backend.chatbot.inference.inference_logger import InferenceLogger
        from backend.chatbot.inference.inference_pipeline import InferencePipeline

        cfg = DbConfig.from_env()
        print("[chat] Loading BGE-M3 embedder...")
        embedder = Embedder()
        print("[chat] Embedder ready.")

        gen = Generator(GeneratorConfig.from_env())
        retriever = Retriever(cfg, embedder)
        builder = PromptBuilder()
        logger = InferenceLogger(cfg)

        pipeline = InferencePipeline(retriever, builder, gen, logger)
        with _chat_lock:
            _chat_state["pipeline"] = pipeline
            _chat_state["status"] = "ready"
            _chat_state["mode"] = gen.mode
        print(f"[chat] Pipeline ready (mode={gen.mode}).")
    except Exception as exc:
        import traceback; traceback.print_exc()
        with _chat_lock:
            _chat_state["status"] = "error"
            _chat_state["error"] = str(exc)

def _ensure_chat_pipeline():
    with _chat_lock:
        st = _chat_state["status"]
        if st in ("ready", "loading", "error"):
            return st
        _chat_state["status"] = "loading"
    threading.Thread(target=_init_chat_pipeline_blocking, daemon=True, name="chat-init").start()
    return "loading"

@app.get("/api/chat/status")
def api_chat_status():
    _ensure_chat_pipeline()
    with _chat_lock:
        return jsonify({
            "status": _chat_state["status"],
            "mode": _chat_state["mode"],
            "error": _chat_state["error"]
        })

@app.post("/api/chat")
def api_chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "Câu hỏi không được để trống."}), 400
    if len(question) > 2000:
        return jsonify({"ok": False, "error": "Câu hỏi quá dài (>2000 ký tự)."}), 400

    status = _ensure_chat_pipeline()
    if status != "ready":
        with _chat_lock:
            err = _chat_state["error"]
        if status == "error":
            return jsonify({"ok": False, "status": "error", "error": err or "Khởi tạo mô hình thất bại."}), 503
        return jsonify({"ok": False, "status": "loading", "message": "Đang khởi tạo mô hình, vui lòng thử lại sau vài giây."}), 503

    pipeline = _chat_state["pipeline"]
    try:
        result = pipeline.answer(question)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"ok": False, "error": f"Lỗi inference: {e}"}), 500

    return jsonify({
        "ok": True,
        "found": result.found,
        "answer": result.answer,
        "latency_ms": result.latency_ms,
        "mode": result.mode,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "doc_name": c.doc_name,
                "doc_code": c.doc_code,
                "dieu": c.dieu,
                "khoan": c.khoan,
                "diem": c.diem,
                "content": c.full_text,
                "similarity": round(c.similarity, 4)
            }
            for c in result.chunks
        ],
    })

@app.post("/api/chat/feedback")
def api_chat_feedback():
    payload = request.get_json(silent=True) or {}
    log_id = payload.get("log_id")
    feedback = payload.get("feedback")  # 1 = thumb up, -1 = thumb down
    if log_id is None or feedback not in (1, -1):
        return jsonify({"ok": False, "error": "Thiếu log_id hoặc feedback không hợp lệ"}), 400
    try:
        with _pg_connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE inference_logs SET user_feedback = %s WHERE id = %s", (feedback, log_id))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/eval/metrics")
def api_eval_metrics():
    eval_path = ROOT / "backend" / "chatbot" / "eval" / "results_baseline.json"
    if not eval_path.is_file():
        return jsonify({"ok": False, "error": "Chưa có kết quả eval. Chạy backend/chatbot/eval/run_baselines.py trước."}), 404
    try:
        import json as _json
        data = _json.loads(eval_path.read_text(encoding = "utf-8"))
        return jsonify({"ok": True, "data": data, "generated_at": eval_path.stat().st_mtime})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "modules": {
            "string_matching": True,
            "text_summarisation": True
        }
    })

APP_PREFIX = os.environ.get("APP_PREFIX", "")
if APP_PREFIX and APP_PREFIX != "/":
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.wrappers import Response as _WerkzeugResponse

    _inner_app = app.wsgi_app

    def _redirect_root(environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path.startswith("/api/"):
            return _inner_app(environ, start_response)
        return _WerkzeugResponse("", status = 302, headers = {"Location": APP_PREFIX + "/"})(environ, start_response)

    app.wsgi_app = DispatcherMiddleware(_redirect_root, {APP_PREFIX: _inner_app})

if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", 9010))
    app.run(host = "0.0.0.0", port = port, debug = False)
