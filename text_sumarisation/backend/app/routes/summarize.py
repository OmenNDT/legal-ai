from flask import Blueprint, jsonify, request
from backend.app.state import AppState
from backend.app.schemas import SummarizeRequest
from backend.evaluate.rouge_scorer import RougeEvaluator
from backend.evaluate.bert_scorer import BertScoreEvaluator

bp = Blueprint("summarize", __name__, url_prefix = "/api/summarize")

# Một bộ đánh giá ROUGE dùng chung
_rouge = RougeEvaluator()
# BertScore lười khởi tạo - load roberta-large ~1.4GB lần đầu
_bertscore: BertScoreEvaluator | None = None

def _get_bertscore() -> BertScoreEvaluator:
    global _bertscore
    if _bertscore is None:
        _bertscore = BertScoreEvaluator(
            lang = "en",
            device = AppState.instance().settings.device(),
            model_type = "roberta-large"
        )
    return _bertscore

# Endpoint chính: chạy pipeline lai trên 1 văn bản
@bp.post("")
def summarize():
    payload = request.get_json(silent = True) or {}
    try:
        req = SummarizeRequest.from_json(payload)
        req.validate()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    state = AppState.instance()
    raw_text = req.text or ""
    reference = ""
    if req.doc_id:
        try:
            contract = state.loader.load_one(req.doc_id)
        except FileNotFoundError:
            return jsonify({"error": "doc_id không tồn tại"}), 404
        raw_text = contract.raw_text
        reference = state.references.get_reference(req.doc_id)

    pipeline = state.get_pipeline(req.extractor, req.use_abstractive)
    result = pipeline.run(raw_text, doc_id = req.doc_id)
    payload_out = result.to_dict()

    # Tính ROUGE + BERTScore nếu có reference (chỉ khi chạy trên doc_id)
    if reference:
        pred = result.abstractive.text if result.abstractive else result.extractive.as_text()
        payload_out["rouge"] = _rouge.score(pred, reference).to_dict()
        payload_out["reference"] = reference
        # BERTScore - đánh giá theo nghĩa
        try:
            p, r, f = _get_bertscore().score([pred], [reference])
            payload_out["bertscore"] = {"precision": p, "recall": r, "f1": f}
        except Exception as e:
            payload_out["bertscore_error"] = str(e)

    return jsonify(payload_out)
