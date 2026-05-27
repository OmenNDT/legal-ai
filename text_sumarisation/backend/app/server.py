from pathlib import Path
from flask import Flask, jsonify, send_from_directory, abort, redirect
from flask_cors import CORS
from backend.config.settings import get_settings
from backend.utils.logger import Logger
from backend.app.state import AppState
from backend.app.routes import summarize as r_summarize
from backend.app.routes import extract as r_extract
from backend.app.routes import documents as r_documents
from backend.app.routes import eval as r_eval
from backend.app import auth as r_auth

# Đường dẫn frontend đã build (vite build -> frontend/dist) với base /legal-ai/
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

# Factory tạo Flask app
def create_app() -> Flask:
    settings = get_settings()
    Logger.setup(settings.LOG_DIR, name = "api")
    app = Flask(__name__)
    # Cho phép frontend dev gọi qua proxy
    CORS(app, resources = {r"/api/*": {"origins": "*"}})

    # Khởi tạo state sớm để bắt lỗi config
    AppState.instance()

    # Đăng ký các blueprint API
    app.register_blueprint(r_summarize.bp)
    app.register_blueprint(r_extract.bp)
    app.register_blueprint(r_documents.bp)
    app.register_blueprint(r_eval.bp)
    app.register_blueprint(r_auth.bp)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "device": settings.device(), "model": settings.BART_MODEL})

    # Phục vụ frontend SPA dưới prefix /legal-ai/ (giống vite base)
    @app.route("/")
    def root_redirect():
        return redirect("/legal-ai/", code = 302)

    @app.route("/legal-ai/")
    def serve_index():
        if not FRONTEND_DIST.exists():
            abort(404, description = "Chưa build frontend. Chạy: cd frontend && npm run build")
        return send_from_directory(FRONTEND_DIST, "index.html")

    @app.route("/legal-ai/<path:path>")
    def serve_static(path: str):
        f = FRONTEND_DIST / path
        if f.is_file():
            return send_from_directory(FRONTEND_DIST, path)
        # SPA fallback: route con không phải file -> trả index.html
        return send_from_directory(FRONTEND_DIST, "index.html")

    return app

# Chạy trực tiếp: python -m backend.app.server
if __name__ == "__main__":
    settings = get_settings()
    app = create_app()
    app.run(host = settings.API_HOST, port = settings.API_PORT, debug = False)