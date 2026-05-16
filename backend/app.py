import os
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

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

from backend.auth import auth_bp
from backend.auth.db import init_pool as init_auth_pool

app.register_blueprint(auth_bp)
try:
    init_auth_pool()
    print("Auth DB pool initialized.")
except Exception as exc:
    print(f"Warning: Auth DB pool not initialized: {exc}")

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
        question=processed,
        top_k=body.get("top_k", 10),
        filters=processed.filters,
    )

    if reasoner is not None:
        augmented = my_rag_pipeline.augmenter.augment_with_kg(
            question=body.get("question", ""),
            retrieval_result=retrieval_result,
            reasoner=reasoner,
            top_k=body.get("top_k_rerank", 5),
        )
    else:
        augmented = my_rag_pipeline.augmenter.augment(
            question=body.get("question", ""),
            retrieval_result=retrieval_result,
            top_k=body.get("top_k_rerank", 5),
        )

    return jsonify({
        "original_question": augmented.original_question,
        "context_text": augmented.context_text,
        "documents_used": len(augmented.documents),
        "token_count": augmented.token_count,
        "strategy": augmented.context_strategy,
        "rerank_scores": augmented.rerank_scores,
    })


@app.post("/rag/generate")
def rag_generate():
    """Phần 4: Sinh câu trả lời từ augmented context."""
    body = request.get_json(force=True)
    if my_rag_pipeline is None:
        return jsonify({"error": "My RAG Pipeline not initialized."})

    req = RAGPipelineRequest(
        question=body.get("question", ""),
        top_k_retrieval=body.get("top_k", 10),
        top_k_rerank=body.get("top_k_rerank", 5),
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
        "latency_ms": response.latency_ms,
    })


@app.post("/rag/pipeline")
def rag_pipeline_endpoint():
    """Full RAG Pipeline (Phần 2→3→4): từ câu hỏi đến câu trả lời."""
    body = request.get_json(force=True)
    if my_rag_pipeline is None:
        return jsonify({"error": "My RAG Pipeline not initialized."})

    req = RAGPipelineRequest(
        question=body.get("question", ""),
        top_k_retrieval=body.get("top_k_retrieval", 10),
        top_k_rerank=body.get("top_k_rerank", 5),
        max_context_tokens=body.get("max_context_tokens", 1024),
        use_reranker=body.get("use_reranker", True),
        return_sources=body.get("return_sources", True),
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
        "latency_ms": response.latency_ms,
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "modules": {
            "pipeline": rag_pipeline is not None,
            "my_rag_pipeline": my_rag_pipeline is not None,
            "search": search_engine is not None,
            "knowledge_graph": kg is not None,
        },
    })

if __name__ == "__main__":
    _startup()
    app.run(host = "0.0.0.0", port = 9010, debug = True)
