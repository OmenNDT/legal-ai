import os
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

from backend.common.config import (
    SEARCH_INDEX_PATH, KG_PATH,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBEDDING_MODEL, CROSS_ENCODER_MODEL,
)
from backend.pipeline.rag_pipeline import RAGPipeline, PipelineConfig
from backend.search.search_engine import LegalSearchEngine
from backend.search.vector_store import VectorStore
from backend.knowledge.graph_builder import LegalKnowledgeGraph
from backend.knowledge.reasoner import LegalReasoner
from backend.knowledge.visualizer import visualize_graph

# My RAG Pipeline (Phần 2-3-4)
from backend.rag_pipeline import (
    RAGPipeline as MyRAGPipeline,
    RAGPipelineRequest,
    MockPreprocessor,
)

app = Flask(__name__)
CORS(app)

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

from backend.auth import auth_bp
from backend.auth.db import init_pool as init_auth_pool

# RAG Extract (from BTMH RAG_Extract)
from backend.rag_extract import rag_bp

app.register_blueprint(auth_bp)
app.register_blueprint(rag_bp)
try:
    init_auth_pool()
    print("Auth DB pool initialized.")
except Exception as exc:
    print(f"Warning: Auth DB pool not initialized: {exc}")

# Initialize RAG Extract database
try:
    from backend.rag_extract.database import init_db as init_rag_db
    init_rag_db()
    print("RAG Extract DB initialized.")
except Exception as exc:
    print(f"Warning: RAG Extract DB not initialized: {exc}")


# ── before_request: skip catch-all for API routes ──
@app.before_request
def _before_request():
    if request.path.startswith('/api/'):
        # Let blueprint routes handle API; if none match, Flask will 404 naturally
        pass


@app.get("/")
def _serve_index():
    if not FRONTEND_DIST.exists():
        return jsonify({"error": "Frontend not built. Run `npm run build` in frontend/."}), 503
    return send_from_directory(str(FRONTEND_DIST), "index.html")


@app.get("/<path:filename>")
def _serve_static(filename: str):
    if not FRONTEND_DIST.exists():
        return jsonify({"error": "Frontend not built."}), 503
    target = FRONTEND_DIST / filename
    if target.is_file():
        return send_from_directory(str(FRONTEND_DIST), filename)
    return send_from_directory(str(FRONTEND_DIST), "index.html")

rag_pipeline: Optional[RAGPipeline] = None
my_rag_pipeline: Optional[MyRAGPipeline] = None
search_engine: Optional[LegalSearchEngine] = None
kg: Optional[LegalKnowledgeGraph] = None
reasoner: Optional[LegalReasoner] = None

def _startup():
    global rag_pipeline, search_engine, kg, reasoner

    config = PipelineConfig(
        llm_provider = LLM_PROVIDER,
        llm_model = LLM_MODEL,
        llm_api_key = LLM_API_KEY,
        llm_base_url = LLM_BASE_URL,
        chroma_host = CHROMA_HOST,
        chroma_port = CHROMA_PORT,
        chroma_collection = CHROMA_COLLECTION,
        embedding_model = EMBEDDING_MODEL,
        cross_encoder_model = CROSS_ENCODER_MODEL,
        kg_path = str(KG_PATH) if Path(KG_PATH).exists() else None,
    )
    rag_pipeline = RAGPipeline(config)

    if Path(SEARCH_INDEX_PATH).exists():
        search_engine = LegalSearchEngine()
        search_engine.index.load(str(SEARCH_INDEX_PATH))
        search_engine.build()

    if Path(KG_PATH).exists():
        kg = LegalKnowledgeGraph()
        kg.load(str(KG_PATH))
        reasoner = LegalReasoner(kg)

    # Initialize my RAG Pipeline (Phần 2-3-4)
    global my_rag_pipeline

    # Vector store + hybrid search
    vector_store = None
    hybrid_search = None
    if search_engine is not None:
        try:
            vector_store = VectorStore(
                collection_name=CHROMA_COLLECTION,
                embedding_model=EMBEDDING_MODEL,
                host=CHROMA_HOST,
                port=CHROMA_PORT,
            )
            from backend.search.hybrid_search import HybridSearch
            hybrid_search = HybridSearch(
                bm25=search_engine.bm25,
                vector_store=vector_store,
                bm25_weight=0.5,
                vector_weight=0.5,
            )
        except Exception as exc:
            print(f"Warning: Could not initialize vector store: {exc}")

    # LLM client (Ollama / OpenAI compatible)
    llm_client = None
    if LLM_API_KEY:
        try:
            from backend.llm.client import LLMClient
            llm_client = LLMClient(
                provider=LLM_PROVIDER,
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
            )
        except Exception as exc:
            print(f"Warning: Could not initialize LLM client: {exc}")

    my_rag_pipeline = MyRAGPipeline(
        search_engine=search_engine,
        reasoner=reasoner,
        vector_store=vector_store,
        hybrid_search=hybrid_search,
        llm_client=llm_client,
    )
    print("My RAG Pipeline initialized (Phần 2-3-4)")

@app.get("/")
def root():
    return jsonify({
        "name": "LegalAI Platform",
        "version": "1.0.0",
        "pipeline_ready": rag_pipeline is not None,
        "search_loaded": search_engine is not None,
        "kg_loaded": kg is not None,
    })

@app.post("/chat")
def chat():
    body = request.get_json(force=True)
    if rag_pipeline is None:
        return jsonify({"error": "Pipeline not initialized."})
    response = rag_pipeline.query(body["question"])
    return jsonify({
        "answer": response.answer,
        "intent": response.intent,
        "domain": response.domain,
        "confidence": response.confidence,
        "entities": response.entities,
        "citations": response.citations,
        "legal_basis": response.legal_basis,
        "conclusion": response.conclusion,
        "recommendation": response.recommendation,
    })


@app.post("/search")
def search():
    body = request.get_json(force=True)
    if search_engine is None:
        return jsonify({"error": "Search engine not loaded. Run build_index.py first."})
    results = search_engine.search(body["query"], top_k=body.get("top_k", 10))
    for r in results:
        r.pop("model", None)
    return jsonify({"query": body["query"], "results": results})


@app.post("/search/autocomplete")
def autocomplete():
    prefix = request.args.get("prefix", "")
    max_results = int(request.args.get("max_results", 10))
    if search_engine is None:
        return jsonify({"error": "Search engine not loaded."})
    return jsonify({"suggestions": search_engine.autocomplete(prefix, max_results)})


@app.post("/search/explain")
def explain():
    query = request.args.get("query", "")
    doc_id = int(request.args.get("doc_id", 0))
    if search_engine is None:
        return jsonify({"error": "Search engine not loaded."})
    return jsonify(search_engine.explain(query, doc_id))

@app.post("/summarize")
def summarize():
    body = request.get_json(force=True)
    from backend.summarizer.summarizer import LegalSummarizer
    result = LegalSummarizer().summarize(
        document=body["document"],
        query=body.get("query", ""),
        top_k=body.get("top_k", 4),
        use_model=False,
    )
    return jsonify({"summary": result["summary"], "selected_indices": result["selected_indices"]})

@app.post("/knowledge/query")
def knowledge_query():
    body = request.get_json(force=True)
    if reasoner is None:
        return jsonify({"error": "Knowledge graph not loaded. Run build_graph.py first."})
    doc_id = body["doc_id"]
    query_type = body.get("query_type", "validity")
    target_id = body.get("target_id")
    if query_type == "validity":
        result = reasoner.check_validity(doc_id)
    elif query_type == "amendments":
        result = reasoner.trace_amendments(doc_id)
    elif query_type == "related":
        result = reasoner.find_related(doc_id)
    elif query_type == "path" and target_id:
        result = reasoner.find_reasoning_path(doc_id, target_id)
    else:
        return jsonify({"error": f"Unknown query_type: {query_type}"})
    return jsonify({
        "question": result.question,
        "answer": result.answer,
        "confidence": result.confidence,
        "reasoning_steps": result.reasoning_steps,
        "evidence": result.evidence,
    })

@app.post("/knowledge/stats")
def knowledge_stats():
    if kg is None:
        return jsonify({"error": "Knowledge graph not loaded."})
    return jsonify(kg.get_stats())

@app.post("/knowledge/visualize")
def knowledge_visualize():
    doc_id = request.args.get("doc_id")
    max_nodes = int(request.args.get("max_nodes", 100))
    if kg is None:
        return jsonify({"error": "Knowledge graph not loaded."})
    output = visualize_graph(kg, output_path="legal_kg.html", max_nodes=max_nodes, focus_node=doc_id)
    return jsonify({"visualization": output})

# ── My RAG Pipeline Endpoints (Phần 2-3-4) ─────────────────

@app.post("/rag/retrieve")
def rag_retrieve():
    """Phần 2: Truy hồi tài liệu liên quan."""
    body = request.get_json(force=True)
    if my_rag_pipeline is None:
        return jsonify({"error": "My RAG Pipeline not initialized."})

    processed = my_rag_pipeline.preprocessor.process(body.get("question", ""))

    result = my_rag_pipeline.retriever.retrieve(
        question=processed,
        top_k=body.get("top_k", 10),
        filters=processed.filters,
    )

    return jsonify({
        "query": result.query,
        "documents": [
            {
                "doc_id": d.doc_id,
                "content": d.content[:300] + "..." if len(d.content) > 300 else d.content,
                "metadata": d.metadata,
                "score": d.score,
                "rank": d.rank,
            }
            for d in result.documents
        ],
        "total_found": result.total_found,
        "method": result.retrieval_method,
        "latency_ms": result.latency_ms,
    })


@app.post("/rag/augment")
def rag_augment():
    """Phần 3: Bổ sung ngữ cảnh từ retrieval results."""
    body = request.get_json(force=True)
    if my_rag_pipeline is None:
        return jsonify({"error": "My RAG Pipeline not initialized."})

    processed = my_rag_pipeline.preprocessor.process(body.get("question", ""))

    retrieval_result = my_rag_pipeline.retriever.retrieve(
        question = processed,
        top_k = body.get("top_k", 10),
        filters = processed.filters
    )

    if reasoner is not None:
        augmented = my_rag_pipeline.augmenter.augment_with_kg(
            question=body.get("question", ""),
            retrieval_result=retrieval_result,
            reasoner=reasoner,
            top_k=body.get("top_k_rerank", 5)
        )
    else:
        augmented = my_rag_pipeline.augmenter.augment(
            question=body.get("question", ""),
            retrieval_result=retrieval_result,
            top_k=body.get("top_k_rerank", 5)
        )

    return jsonify({
        "original_question": augmented.original_question,
        "context_text": augmented.context_text,
        "documents_used": len(augmented.documents),
        "token_count": augmented.token_count,
        "strategy": augmented.context_strategy,
        "rerank_scores": augmented.rerank_scores
    })


@app.post("/rag/generate")
def rag_generate():
    """Phần 4: Sinh câu trả lời từ augmented context."""
    body = request.get_json(force=True)
    if my_rag_pipeline is None:
        return jsonify({"error": "My RAG Pipeline not initialized."})

    req = RAGPipelineRequest(
        question = body.get("question", ""),
        top_k_retrieval = body.get("top_k", 10),
        top_k_rerank = body.get("top_k_rerank", 5),
    )
    response = my_rag_pipeline.run(req)

    return jsonify({
        "answer": response.answer,
        "confidence": response.confidence,
        "sources": [
            {
                "doc_id": s.doc_id,
                "name": s.doc_name,
                "excerpt": s.excerpt[:200] + "..." if len(s.excerpt) > 200 else s.excerpt,
                "relevance": s.relevance_score,
            }
            for s in response.sources
        ],
        "reasoning": response.reasoning,
        "latency_ms": response.latency_ms
    })


@app.post("/rag/pipeline")
def rag_pipeline_endpoint():
    """Full RAG Pipeline (Phần 2→3→4): từ câu hỏi đến câu trả lời."""
    body = request.get_json(force=True)
    if my_rag_pipeline is None:
        return jsonify({"error": "My RAG Pipeline not initialized."})

    req = RAGPipelineRequest(
        question = body.get("question", ""),
        top_k_retrieval = body.get("top_k_retrieval", 10),
        top_k_rerank = body.get("top_k_rerank", 5),
        max_context_tokens = body.get("max_context_tokens", 1024),
        use_reranker = body.get("use_reranker", True),
        return_sources = body.get("return_sources", True)
    )
    response = my_rag_pipeline.run(req)

    return jsonify({
        "answer": response.answer,
        "confidence": response.confidence,
        "sources": [
            {
                "doc_id": s.doc_id,
                "name": s.doc_name,
                "excerpt": s.excerpt[:200] + "..." if len(s.excerpt) > 200 else s.excerpt,
                "relevance": s.relevance_score,
            }
            for s in response.sources
        ],
        "reasoning": response.reasoning,
        "retrieval": {
            "method": response.retrieval_info.retrieval_method,
            "total_found": response.retrieval_info.total_found,
            "latency_ms": response.retrieval_info.latency_ms,
        },
        "latency_ms": response.latency_ms
    })


def _run_matcher(matcher_cls, algo_key):
    import time
    payload = request.get_json(silent = True) or {}
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
    from datetime import datetime
    import re as _re

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "") or ""
    pattern = payload.get("pattern", "") or ""
    case_sensitive = bool(payload.get("case_sensitive", False))
    positions = payload.get("positions") or []
    occurrences = payload.get("occurrences")
    if occurrences is None:
        occurrences = len(positions)
    complexities = payload.get("complexities") or {}  # {naive, kmp, boyer_moore} -> worst-case Big-O string
    comparisons = payload.get("comparisons") or {}  # same keys -> int

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

    out_dir = Path(__file__).resolve().parent / "string_matching" / "result"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_pattern = _re.sub(r"[^a-zA-Z0-9_-]+", "_", pattern.strip())[:30] or "pattern"
    filename = f"result_{ts}_{safe_pattern}.txt"
    out_path = out_dir / filename
    out_path.write_text(content, encoding="utf-8")

    return jsonify({"ok": True, "filename": filename, "path": str(out_path), "content": content})


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "modules": {
            "pipeline": rag_pipeline is not None,
            "my_rag_pipeline": my_rag_pipeline is not None,
            "search": search_engine is not None,
            "knowledge_graph": kg is not None
        },
    })

APP_PREFIX = os.environ.get("APP_PREFIX", "/legal-ai")

if APP_PREFIX and APP_PREFIX != "/":
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.wrappers import Response

    _original_wsgi_app = app.wsgi_app

    def _redirect_root(environ, start_response):
        path = environ.get('PATH_INFO', '')
        if path.startswith('/api/'):
            return _original_wsgi_app(environ, start_response)
        return Response("", status=302, headers={"Location": APP_PREFIX + "/"})(environ, start_response)

    app.wsgi_app = DispatcherMiddleware(_redirect_root, {APP_PREFIX: _original_wsgi_app})

if __name__ == "__main__":
    _startup()
    app.run(host = "0.0.0.0", port = 9010, debug = True)
