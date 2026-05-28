import os
import sys
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

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
