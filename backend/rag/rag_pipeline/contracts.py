"""Pydantic data contracts for RAG Pipeline (Phần 2-3-4).

Định nghĩa interface giữa các phần để đảm bảo tương thích
khi ghép nối với phần 1 (Preprocessing) và phần 5 (Postprocessing).
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Input từ Phần 1 (Tiền xử lý) ──────────────────────

class ExtractedEntity(BaseModel):
    """Entity extracted from user question."""
    text: str
    label: str  # e.g., "LUAT", "DIEU", "NGHI_DINH"
    start: int
    end: int


class ProcessedQuestion(BaseModel):
    """Câu hỏi đã qua tiền xử lý từ Phần 1.

    Đây là input đầu vào cho Phần 2 (Retrieval).
    """
    raw_text: str = Field(..., description="Câu hỏi gốc của người dùng")
    segmented_text: str = Field(..., description="Câu hỏi đã word-segment")
    intent: str = Field(..., description="Intent đã phân loại")
    intent_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict, description="Metadata filters từ entities")


# ── Output Phần 2 (Retrieval) ───────────────────────────

class RetrievedDocument(BaseModel):
    """Một tài liệu được truy hồi."""
    doc_id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    score: float = Field(..., ge=0.0, description="Retrieval score (BM25 hoặc similarity)")
    rank: int = Field(..., ge=1, description="Thứ hạng sau ranking")


class RetrievalResult(BaseModel):
    """Kết quả truy hồi từ Phần 2.

    Là input cho Phần 3 (Augment).
    """
    query: str
    documents: list[RetrievedDocument]
    total_found: int
    retrieval_method: str = Field(default="bm25", description="bm25 | vector | hybrid")
    latency_ms: float = Field(default=0.0)


# ── Output Phần 3 (Augment) ─────────────────────────────

class AugmentedContext(BaseModel):
    """Ngữ cảnh đã được bổ sung từ Phần 3.

    Là input cho Phần 4 (Generation).
    """
    original_question: str
    context_text: str = Field(..., description="Văn bản ngữ cảnh đã ghép")
    documents: list[RetrievedDocument]
    rerank_scores: list[float] = Field(default_factory=list)
    token_count: int = Field(default=0)
    context_strategy: str = Field(default="concat", description="concat | hierarchical | summary")


# ── Output Phần 4 (Generation) ────────────────────────

class Citation(BaseModel):
    """Trích dẫn nguồn trong câu trả lời."""
    doc_id: str
    doc_name: str = ""
    excerpt: str = ""
    relevance_score: float = 0.0


class GeneratedAnswer(BaseModel):
    """Câu trả lời đã sinh từ Phần 4.

    Là input cho Phần 5 (Postprocessing).
    """
    answer_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    generation_method: str = Field(default="phobert", description="phobert | llm | extractive")
    latency_ms: float = Field(default=0.0)


# ── Full Pipeline Request/Response ──────────────────────

class RAGPipelineRequest(BaseModel):
    """Request cho toàn bộ pipeline 2→3→4."""
    question: str
    top_k_retrieval: int = Field(default=10, ge=1, le=50)
    top_k_rerank: int = Field(default=5, ge=1, le=20)
    max_context_tokens: int = Field(default=1024, ge=256, le=4096)
    use_reranker: bool = True
    return_sources: bool = True


class RAGPipelineResponse(BaseModel):
    """Response từ toàn bộ pipeline 2→3→4."""
    answer: str
    confidence: float
    sources: list[Citation]
    reasoning: list[str]
    retrieval_info: RetrievalResult
    latency_ms: float
