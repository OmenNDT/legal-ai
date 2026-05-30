from flask import Blueprint, jsonify, request
from backend.app.state import AppState
from backend.app.schemas import SummarizeRequest

bp = Blueprint("extract", __name__, url_prefix = "/api/extract")

# Chỉ chạy bước extractive - nhanh, không cần GPU
@bp.post("")
def extract_only():
    payload = request.get_json(silent = True) or {}
    payload["use_abstractive"] = False
    try:
        req = SummarizeRequest.from_json(payload)
        req.validate()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    state = AppState.instance()
    raw_text = req.text
    if req.doc_id:
        try:
            raw_text = state.loader.load_one(req.doc_id).raw_text
        except FileNotFoundError:
            return jsonify({"error": "doc_id không tồn tại"}), 404

    pipeline = state.get_pipeline(req.extractor, use_abstractive = False)
    result = pipeline.run(raw_text, doc_id = req.doc_id)
    return jsonify(result.to_dict())
