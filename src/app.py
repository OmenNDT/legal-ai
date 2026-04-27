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


app = FastAPI(title="LegalAI Platform", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global state (loaded on startup) ──────────────────
search_engine: Optional[LegalSearchEngine] = None
kg: Optional[LegalKnowledgeGraph] = None
reasoner: Optional[LegalReasoner] = None


@app.on_event("startup")
async def startup():
    """Load models and indices on startup."""
    global search_engine, kg, reasoner

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


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)