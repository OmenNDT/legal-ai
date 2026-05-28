from flask import Blueprint, jsonify, request, abort
from backend.app.state import AppState

bp = Blueprint("documents", __name__, url_prefix = "/api/documents")

# Trả về danh sách doc_id, có phân trang nhẹ
@bp.get("")
def list_documents():
    state = AppState.instance()
    ids = state.loader.list_ids()
    q = (request.args.get("q") or "").lower().strip()
    if q:
        ids = [i for i in ids if q in i.lower()]
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 50))
    start = (page - 1) * page_size
    end = start + page_size
    return jsonify({
        "total": len(ids),
        "page": page,
        "page_size": page_size,
        "items": ids[start:end],
    })

# Trả về nội dung của 1 doc + reference (nếu có)
@bp.get("/<doc_id>")
def get_document(doc_id: str):
    state = AppState.instance()
    try:
        contract = state.loader.load_one(doc_id)
    except FileNotFoundError:
        abort(404, description = "Không tìm thấy doc_id")
    reference = state.references.get_reference(doc_id)
    return jsonify({
        "doc_id": doc_id,
        "word_count": contract.word_count,
        "text": contract.raw_text,
        "reference": reference,
    })
