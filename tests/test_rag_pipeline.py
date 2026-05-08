"""Pytest suite for RAG Pipeline (Phần 2-3-4)."""

import pytest

from src.rag_pipeline import (
    MockPreprocessor,
    MockPostprocessor,
    LegalRetriever,
    ContextAugmenter,
    LegalAnswerGenerator,
    RAGPipeline,
    RAGPipelineRequest,
)
from src.rag_pipeline.contracts import (
    GeneratedAnswer,
    Citation,
    RetrievedDocument,
    RetrievalResult,
    AugmentedContext,
)


class TestMockPreprocessor:
    def test_process_returns_processed_question(self):
        pre = MockPreprocessor()
        result = pre.process("Luật Kế toán 2015 còn hiệu lực không?")
        assert result.raw_text == "Luật Kế toán 2015 còn hiệu lực không?"
        assert result.intent is not None
        assert isinstance(result.entities, list)
        assert isinstance(result.filters, dict)

    def test_intent_detection_hieu_luc(self):
        pre = MockPreprocessor()
        result = pre.process("Luật này còn hiệu lực không?")
        assert result.intent == "hoi_hieu_luc"
        assert result.intent_confidence > 0.5

    def test_entity_extraction_luat(self):
        pre = MockPreprocessor()
        result = pre.process("Luật Kế toán 2015 quy định gì?")
        entities = [(e.text, e.label) for e in result.entities]
        assert any(label == "LUAT" for _, label in entities)


class TestMockPostprocessor:
    def test_format_returns_dict(self):
        answer = GeneratedAnswer(
            answer_text="Theo Luật Kế toán 2015, văn bản này còn hiệu lực.",
            confidence=0.92,
            citations=[
                Citation(doc_id="doc_1", doc_name="Luật Kế toán 2015", excerpt="Điều 1...", relevance_score=0.95),
            ],
            reasoning_steps=["Truy hồi tài liệu", "Kiểm tra hiệu lực"],
            generation_method="extractive",
            latency_ms=150.0,
        )
        post = MockPostprocessor()
        result = post.format(answer, return_markdown=True)
        assert "answer" in result
        assert "confidence" in result
        assert result["confidence"] == 0.92
        assert "markdown" in result


class TestLegalRetriever:
    def test_retrieve_without_search_engine(self, sample_processed_question):
        retriever = LegalRetriever(search_engine=None)
        result = retriever.retrieve(sample_processed_question, top_k=5)
        assert result.total_found == 0
        assert result.documents == []
        assert result.latency_ms >= 0

    def test_retrieve_with_mock_engine(self, mock_search_engine, sample_processed_question):
        retriever = LegalRetriever(search_engine=mock_search_engine)
        result = retriever.retrieve(sample_processed_question, top_k=3)
        assert len(result.documents) <= 3
        assert result.latency_ms >= 0
        if result.documents:
            assert all(d.score >= 0 for d in result.documents)
            assert all(d.rank >= 1 for d in result.documents)

    def test_retrieve_with_expansion(self, mock_search_engine, sample_processed_question):
        from src.rag_pipeline.query_expansion import expand_query
        expanded = expand_query(sample_processed_question.raw_text, sample_processed_question.entities)
        assert len(expanded) >= 1
        assert sample_processed_question.raw_text in expanded


class TestContextAugmenter:
    def test_augment_token_limit(self, sample_retrieval_result):
        augmenter = ContextAugmenter(max_tokens=256, use_reranker=False)
        augmented = augmenter.augment(
            question="Luật Kế toán 2015 còn hiệu lực không?",
            retrieval_result=sample_retrieval_result,
            top_k=5,
        )
        assert augmented.token_count <= 256
        assert len(augmented.documents) <= len(sample_retrieval_result.documents)
        assert augmented.context_strategy in ("concat", "concat+kg")

    def test_augment_with_empty_results(self):
        augmenter = ContextAugmenter(max_tokens=512)
        empty_result = RetrievalResult(
            query="test", documents=[], total_found=0, retrieval_method="bm25"
        )
        augmented = augmenter.augment(question="test?", retrieval_result=empty_result, top_k=5)
        assert augmented.context_text == "Không tìm thấy tài liệu liên quan."
        assert augmented.documents == []


class TestLegalAnswerGenerator:
    def test_extractive_generation(self, sample_augmented_context):
        generator = LegalAnswerGenerator(generation_mode="extractive")
        answer = generator.generate(sample_augmented_context)
        assert answer.answer_text != ""
        assert 0.0 <= answer.confidence <= 1.0
        assert answer.generation_method == "extractive"
        assert answer.latency_ms >= 0
        assert isinstance(answer.citations, list)
        assert isinstance(answer.reasoning_steps, list)

    def test_llm_fallback_when_no_client(self, sample_augmented_context):
        generator = LegalAnswerGenerator(generation_mode="llm", llm_client=None)
        answer = generator.generate(sample_augmented_context)
        assert answer.answer_text != ""
        # Should fallback to extractive since no LLM client
        assert answer.generation_method == "llm"

    def test_llm_generation(self, sample_augmented_context, mock_llm_client):
        generator = LegalAnswerGenerator(generation_mode="llm", llm_client=mock_llm_client)
        answer = generator.generate(sample_augmented_context)
        assert answer.answer_text != ""
        assert "Căn cứ pháp lý" in answer.answer_text or "Kết luận" in answer.answer_text or "Lưu ý" in answer.answer_text
        assert answer.confidence == 0.85
        assert answer.generation_method == "llm"


class TestRAGPipeline:
    def test_full_pipeline_with_fallback(self):
        pipeline = RAGPipeline(search_engine=None, reasoner=None)
        request = RAGPipelineRequest(
            question="Luật Kế toán 2015 còn hiệu lực không?",
            top_k_retrieval=5,
            top_k_rerank=3,
        )
        response = pipeline.run(request)
        assert response.answer != ""
        assert 0.0 <= response.confidence <= 1.0
        assert isinstance(response.sources, list)
        assert isinstance(response.reasoning, list)
        assert response.latency_ms >= 0
        assert response.retrieval_info is not None

    def test_full_pipeline_with_mock_search(self, mock_search_engine):
        pipeline = RAGPipeline(
            search_engine=mock_search_engine,
            reasoner=None,
        )
        request = RAGPipelineRequest(
            question="Luật Kế toán 2015 còn hiệu lực không?",
            top_k_retrieval=3,
            top_k_rerank=2,
        )
        response = pipeline.run(request)
        assert response.answer != ""
        assert len(response.sources) > 0
        assert response.latency_ms > 0


class TestQueryExpansion:
    def test_expand_query_basic(self):
        from src.rag_pipeline.query_expansion import expand_query
        from src.rag_pipeline.contracts import ExtractedEntity
        entities = [ExtractedEntity(text="Luật Kế toán 2015", label="LUAT", start=0, end=18)]
        result = expand_query("phạt tù là gì?", entities)
        assert len(result) >= 1
        assert "phạt tù là gì?" in result

    def test_detect_domain(self):
        from src.rag_pipeline.query_expansion import detect_domain
        assert detect_domain("hợp đồng lao động") == "lao_dong"
        assert detect_domain("thuế doanh nghiệp") == "thuong_mai"
        assert detect_domain("abc xyz") == "general"
