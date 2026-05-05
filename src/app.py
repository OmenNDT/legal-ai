import os
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.common.config import (
    SEARCH_INDEX_PATH, KG_PATH,
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBEDDING_MODEL, CROSS_ENCODER_MODEL,
)
from src.pipeline.rag_pipeline import RAGPipeline, PipelineConfig
from src.search.search_engine import LegalSearchEngine
from src.knowledge.graph_builder import LegalKnowledgeGraph
from src.knowledge.reasoner import LegalReasoner
from src.knowledge.visualizer import visualize_graph

app = Flask(__name__)
CORS(app)

rag_pipeline: Optional[RAGPipeline] = None
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
    from src.summarizer.summarizer import LegalSummarizer
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

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    _startup()
    app.run(host = "0.0.0.0", port = 8000, debug = True)
