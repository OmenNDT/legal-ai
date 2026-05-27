from flask import Blueprint, jsonify, request
from backend.evaluate.runner import EvalRunner

bp = Blueprint("eval", __name__, url_prefix = "/api/eval")

# Chạy benchmark trên toàn bộ (hoặc limit) doc
@bp.post("/run")
def run_eval():
    body = request.get_json(silent = True) or {}
    extractor = (body.get("extractor") or "textrank").lower()
    use_abstractive = bool(body.get("use_abstractive", True))
    limit = body.get("limit")
    doc_ids = body.get("doc_ids")
    runner = EvalRunner(extractor_name = extractor, use_abstractive = use_abstractive)
    result = runner.run(limit = limit, doc_ids = doc_ids)
    # Trả về phần tóm tắt để khỏi nặng response, chi tiết đã lưu vào file
    return jsonify({
        "extractor": result["extractor"],
        "use_abstractive": result["use_abstractive"],
        "num_docs": result["num_docs"],
        "average": result["average"],
        "preview": result["per_doc"][:5],
    })
