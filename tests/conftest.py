"""Pytest fixtures for RAG Pipeline tests."""

import pytest

from src.rag_pipeline.contracts import (
    ExtractedEntity,
    RetrievedDocument,
    RetrievalResult,
    AugmentedContext,
    ProcessedQuestion,
)


class MockLLMClient:
    """Mock LLM client that returns a fixed structured answer."""

    def __init__(self, response: str = ""):
        self._response = response or (
            "**Căn cứ pháp lý:**\n"
            "Theo Luật Kế toán 2015.\n"
            "**Phân tích:**\n"
            "Văn bản này còn hiệu lực.\n"
            "**Kết luận:**\n"
            "Còn hiệu lực.\n"
            "**Khuyến nghị:**\n"
            "Tham khảo thêm Nghị định 129/2018."
        )

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048, temperature: float = 0.1) -> str:
        return self._response


class MockSearchEngine:
    """Mock LegalSearchEngine with BM25 search."""

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        return [
            {
                "doc_id": f"doc_{i}",
                "content": f"Sample legal document content for {query}.",
                "title": f"Luật mẫu {i}",
                "type": "luat",
                "score": 0.95 - (i * 0.05),
            }
            for i in range(min(top_k, 3))
        ]

    def autocomplete(self, prefix: str, max_results: int = 10) -> list[str]:
        return [f"{prefix}_suggestion_{i}" for i in range(min(max_results, 3))]


@pytest.fixture
def mock_llm_client():
    return MockLLMClient()


@pytest.fixture
def mock_search_engine():
    return MockSearchEngine()


@pytest.fixture
def sample_documents():
    return [
        RetrievedDocument(
            doc_id="doc_1",
            content="Luật Kế toán 2015 quy định về chế độ kế toán.",
            metadata={"doc_type": "luat", "name": "Luật Kế toán 2015"},
            score=0.95,
            rank=1,
        ),
        RetrievedDocument(
            doc_id="doc_2",
            content="Nghị định 129/2018/NĐ-CP hướng dẫn Luật Kế toán.",
            metadata={"doc_type": "nghi_dinh", "name": "Nghị định 129/2018"},
            score=0.85,
            rank=2,
        ),
    ]


@pytest.fixture
def sample_retrieval_result(sample_documents):
    return RetrievalResult(
        query="Luật Kế toán 2015 còn hiệu lực không?",
        documents=sample_documents,
        total_found=2,
        retrieval_method="bm25",
    )


@pytest.fixture
def sample_augmented_context(sample_documents):
    return AugmentedContext(
        original_question="Luật Kế toán 2015 còn hiệu lực không?",
        context_text="Câu hỏi: Luật Kế toán 2015 còn hiệu lực không?\n\n[luat] Luật Kế toán 2015\nLuật Kế toán 2015 quy định về chế độ kế toán.",
        documents=sample_documents,
        token_count=50,
        context_strategy="concat",
    )


@pytest.fixture
def sample_processed_question():
    return ProcessedQuestion(
        raw_text="Luật Kế toán 2015 còn hiệu lực không?",
        segmented_text="Luật Kế toán 2015 còn hiệu lực không ?",
        intent="hoi_hieu_luc",
        intent_confidence=0.92,
        entities=[
            ExtractedEntity(text="Luật Kế toán 2015", label="LUAT", start=0, end=18),
        ],
        filters={"doc_type": "law"},
    )
