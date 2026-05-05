"""FastAPI backend wiring all 4 LegalAI modules."""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.common.config import (
    SEARCH_INDEX_PATH, KG_PATH,
    INTENT_MODEL_DIR, NER_MODEL_DIR, SCORER_MODEL_DIR,
)

# LoRA checkpoint for Luật Kế toán 2025
LORA_CHECKPOINT_PATH = "data/models/lora_ke_toan/best_model.pt"
from src.search.search_engine import LegalSearchEngine
from src.knowledge.graph_builder import LegalKnowledgeGraph
from src.knowledge.reasoner import LegalReasoner
from src.knowledge.visualizer import visualize_graph, visualize_amendment_chain

# RAG Pipeline (Phần 2-3-4)
from src.rag_pipeline import (
    RAGPipeline,
    RAGPipelineRequest,
    RAGPipelineResponse,
    MockPreprocessor,
    MockPostprocessor,
)


app = FastAPI(title="LegalAI Platform", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global state (loaded on startup) ──────────────────
search_engine: Optional[LegalSearchEngine] = None
kg: Optional[LegalKnowledgeGraph] = None
reasoner: Optional[LegalReasoner] = None
rag_pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup():
    """Load models and indices on startup."""
    global search_engine, kg, reasoner, rag_pipeline

    # Load search index
    if Path(SEARCH_INDEX_PATH).exists():
        search_engine = LegalSearchEngine()
        search_engine.index.load(str(SEARCH_INDEX_PATH))
        search_engine.build()
        print(f"Search index loaded: {search_engine.index.doc_count} docs")

    # Load knowledge graph
    if Path(KG_PATH).exists():
        kg = LegalKnowledgeGraph()
        kg.load(str(KG_PATH))
        reasoner = LegalReasoner(kg)
        print(f"Knowledge graph loaded: {kg.node_count} nodes, {kg.edge_count} edges")

    # Initialize RAG Pipeline (Phần 2-3-4)
    rag_pipeline = RAGPipeline(
        search_engine=search_engine,
        reasoner=reasoner,
    )
    print("RAG Pipeline initialized (Phần 2-3-4)")


# ── Request models ─────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 10

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class SummarizeRequest(BaseModel):
    document: str
    query: str = ""
    top_k: int = 4

class KGQueryRequest(BaseModel):
    doc_id: str
    query_type: str = "validity"  # validity | amendments | related | path
    target_id: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "LegalAI Platform",
        "modules": ["chatbot", "search", "summarizer", "knowledge"],
        "search_loaded": search_engine is not None,
        "kg_loaded": kg is not None,
    }


@app.post("/search")
def search(req: SearchRequest):
    """Search for legal documents."""
    if search_engine is None:
        return {"error": "Search engine not loaded. Run build_index.py first."}
    results = search_engine.search(req.query, top_k=req.top_k)
    # Remove non-serializable model objects
    for r in results:
        r.pop("model", None)
    return {"query": req.query, "results": results}


@app.post("/search/autocomplete")
def autocomplete(prefix: str, max_results: int = 10):
    """Get autocomplete suggestions."""
    if search_engine is None:
        return {"error": "Search engine not loaded."}
    return {"suggestions": search_engine.autocomplete(prefix, max_results)}


@app.post("/search/explain")
def explain(query: str, doc_id: int):
    """Explain BM25 scoring for a document."""
    if search_engine is None:
        return {"error": "Search engine not loaded."}
    return search_engine.explain(query, doc_id)


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    """Summarize a legal document."""
    from src.summarizer.summarizer import LegalSummarizer
    summarizer = LegalSummarizer()
    result = summarizer.summarize(
        document=req.document,
        query=req.query,
        top_k=req.top_k,
        use_model=False,  # Use TF-IDF fallback until model is trained
    )
    return {"summary": result["summary"], "selected_indices": result["selected_indices"]}


@app.post("/knowledge/query")
def knowledge_query(req: KGQueryRequest):
    """Query the legal knowledge graph."""
    if reasoner is None:
        return {"error": "Knowledge graph not loaded. Run build_graph.py first."}

    if req.query_type == "validity":
        result = reasoner.check_validity(req.doc_id)
    elif req.query_type == "amendments":
        result = reasoner.trace_amendments(req.doc_id)
    elif req.query_type == "related":
        result = reasoner.find_related(req.doc_id)
    elif req.query_type == "path" and req.target_id:
        result = reasoner.find_reasoning_path(req.doc_id, req.target_id)
    else:
        return {"error": f"Unknown query_type: {req.query_type}"}

    return {
        "question": result.question,
        "answer": result.answer,
        "confidence": result.confidence,
        "reasoning_steps": result.reasoning_steps,
        "evidence": result.evidence,
    }


@app.post("/knowledge/stats")
def knowledge_stats():
    """Get knowledge graph statistics."""
    if kg is None:
        return {"error": "Knowledge graph not loaded."}
    return kg.get_stats()


@app.post("/knowledge/visualize")
def knowledge_visualize(doc_id: Optional[str] = None, max_nodes: int = 100):
    """Generate interactive visualization of the knowledge graph."""
    if kg is None:
        return {"error": "Knowledge graph not loaded."}
    output = visualize_graph(kg, output_path="legal_kg.html", max_nodes=max_nodes, focus_node=doc_id)
    return {"visualization": output}


@app.post("/chat")
def chat(req: QueryRequest):
    """Full chatbot pipeline: intent → NER → search → summarize → KG → response."""
    from src.chatbot.pipeline import LegalChatbot

    lora_path = LORA_CHECKPOINT_PATH if Path(LORA_CHECKPOINT_PATH).exists() else None
    chatbot = LegalChatbot(
        search_engine=search_engine,
        knowledge_graph=kg,
        lora_checkpoint_path=lora_path,
    )
    response = chatbot.ask(req.question, top_k_docs=req.top_k)

    return {
        "answer": response.answer,
        "intent": response.intent,
        "confidence": response.confidence,
        "entities": response.entities,
        "sources": response.sources,
        "summary": response.summary,
        "reasoning": response.reasoning,
    }


# ── RAG Pipeline Endpoints (Phần 2-3-4) ─────────────────

@app.post("/rag/retrieve")
def rag_retrieve(req: QueryRequest):
    """Phần 2: Truy hồi tài liệu liên quan.

    Input: câu hỏi thô
    Output: danh sách tài liệu được xếp hạng
    """
    if rag_pipeline is None:
        return {"error": "RAG Pipeline not initialized."}

    # Debug: check if search_engine is available
    if rag_pipeline.retriever.search_engine is None:
        return {"error": "Search engine not available in retriever"}

    from src.rag_pipeline import MockPreprocessor
    preprocessor = MockPreprocessor()
    processed = preprocessor.process(req.question)

    # Debug info
    debug_info = {
        "question": req.question,
        "segmented": processed.segmented_text,
        "intent": processed.intent,
        "filters": processed.filters,
        "search_engine_doc_count": rag_pipeline.retriever.search_engine.index.doc_count if rag_pipeline.retriever.search_engine else 0,
    }

    result = rag_pipeline.retriever.retrieve(
        question=processed,
        top_k=req.top_k,
        filters=processed.filters,
    )

    return {
        "query": result.query,
        "debug": debug_info,
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
    }


@app.post("/rag/augment")
def rag_augment(req: QueryRequest):
    """Phần 3: Bổ sung ngữ cảnh từ retrieval results.

    Input: câu hỏi thô
    Output: augmented context đã rerank
    """
    if rag_pipeline is None:
        return {"error": "RAG Pipeline not initialized."}

    from src.rag_pipeline import MockPreprocessor
    preprocessor = MockPreprocessor()
    processed = preprocessor.process(req.question)

    retrieval_result = rag_pipeline.retriever.retrieve(
        question=processed,
        top_k=req.top_k,
        filters=processed.filters,
    )

    if reasoner is not None:
        augmented = rag_pipeline.augmenter.augment_with_kg(
            question=req.question,
            retrieval_result=retrieval_result,
            reasoner=reasoner,
            top_k=5,
        )
    else:
        augmented = rag_pipeline.augmenter.augment(
            question=req.question,
            retrieval_result=retrieval_result,
            top_k=5,
        )

    return {
        "original_question": augmented.original_question,
        "context_text": augmented.context_text,
        "documents_used": len(augmented.documents),
        "token_count": augmented.token_count,
        "strategy": augmented.context_strategy,
        "rerank_scores": augmented.rerank_scores,
    }


@app.post("/rag/generate")
def rag_generate(req: QueryRequest):
    """Phần 4: Sinh câu trả lời từ augmented context.

    Input: câu hỏi thô
    Output: câu trả lời với citations
    """
    if rag_pipeline is None:
        return {"error": "RAG Pipeline not initialized."}

    request = RAGPipelineRequest(
        question=req.question,
        top_k_retrieval=req.top_k,
        top_k_rerank=5,
    )
    response = rag_pipeline.run(request)

    return {
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
    }


@app.post("/rag/pipeline")
def rag_pipeline_endpoint(req: RAGPipelineRequest):
    """Full RAG Pipeline (Phần 2→3→4): từ câu hỏi đến câu trả lời.

    Đây là endpoint chính cho phần của bạn.
    Input: câu hỏi + config
    Output: câu trả lời đầy đủ với sources và reasoning
    """
    if rag_pipeline is None:
        return {"error": "RAG Pipeline not initialized. Run build_index.py first."}

    response = rag_pipeline.run(req)

    return {
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
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modules": {
            "search": search_engine is not None,
            "knowledge_graph": kg is not None,
            "rag_pipeline": rag_pipeline is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=9000, reload=True)