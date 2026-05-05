"""Test script cho RAG Pipeline (Phần 2-3-4).

Chạy: python test_rag_pipeline.py
"""

import sys
sys.path.insert(0, "/home/sontn/Projects/legal-ai")

from src.rag_pipeline import (
    MockPreprocessor,
    MockPostprocessor,
    LegalRetriever,
    ContextAugmenter,
    LegalAnswerGenerator,
    RAGPipeline,
    RAGPipelineRequest,
)


def test_mock_preprocessor():
    print("=" * 50)
    print("TEST 1: MockPreprocessor (Phần 1 mock)")
    print("=" * 50)

    pre = MockPreprocessor()
    questions = [
        "Luật Kế toán 2015 còn hiệu lực không?",
        "Thủ tục đăng ký kinh doanh theo Nghị định 01/2021?",
        "Định nghĩa tài sản cố định là gì?",
    ]

    for q in questions:
        result = pre.process(q)
        print(f"\nQ: {result.raw_text}")
        print(f"  Intent: {result.intent} (confidence: {result.intent_confidence})")
        print(f"  Entities: {[(e.text, e.label) for e in result.entities]}")
        print(f"  Filters: {result.filters}")

    print("\n✓ MockPreprocessor OK\n")


def test_mock_postprocessor():
    print("=" * 50)
    print("TEST 2: MockPostprocessor (Phần 5 mock)")
    print("=" * 50)

    from src.rag_pipeline.contracts import GeneratedAnswer, Citation

    answer = GeneratedAnswer(
        answer_text="Theo Luật Kế toán 2015, văn bản này còn hiệu lực.",
        confidence=0.92,
        citations=[
            Citation(doc_id="doc_1", doc_name="Luật Kế toán 2015", excerpt="Điều 1...", relevance_score=0.95),
        ],
        reasoning_steps=["Truy hồi tài liệu", "Kiểm tra hiệu lực", "Tổng hợp câu trả lời"],
        generation_method="extractive",
        latency_ms=150.0,
    )

    post = MockPostprocessor()
    result = post.format(answer, return_markdown=True)

    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources: {len(result.get('sources', []))}")
    print(f"Markdown preview:\n{result['markdown'][:300]}...")

    print("\n✓ MockPostprocessor OK\n")


def test_retriever_without_search_engine():
    print("=" * 50)
    print("TEST 3: LegalRetriever (Phần 2) - without search engine")
    print("=" * 50)

    retriever = LegalRetriever(search_engine=None)
    pre = MockPreprocessor()
    processed = pre.process("Luật Kế toán 2015 còn hiệu lực không?")

    result = retriever.retrieve(processed, top_k=5)

    print(f"Query: {result.query}")
    print(f"Method: {result.retrieval_method}")
    print(f"Documents found: {result.total_found}")
    print(f"Latency: {result.latency_ms:.2f}ms")

    print("\n✓ LegalRetriever OK (fallback mode)\n")


def test_augmenter():
    print("=" * 50)
    print("TEST 4: ContextAugmenter (Phần 3)")
    print("=" * 50)

    from src.rag_pipeline.contracts import RetrievedDocument, RetrievalResult

    augmenter = ContextAugmenter(max_tokens=512, use_reranker=False)

    # Tạo mock retrieval result
    docs = [
        RetrievedDocument(
            doc_id="doc_1",
            content="Luật Kế toán 2015 quy định về chế độ kế toán. Điều 1: Phạm vi điều chỉnh...",
            metadata={"doc_type": "luat", "name": "Luật Kế toán 2015"},
            score=0.95,
            rank=1,
        ),
        RetrievedDocument(
            doc_id="doc_2",
            content="Nghị định 129/2018/NĐ-CP hướng dẫn Luật Kế toán...",
            metadata={"doc_type": "nghi_dinh", "name": "Nghị định 129/2018"},
            score=0.85,
            rank=2,
        ),
    ]

    retrieval = RetrievalResult(
        query="Luật Kế toán 2015 còn hiệu lực không?",
        documents=docs,
        total_found=2,
        retrieval_method="bm25",
    )

    augmented = augmenter.augment(
        question="Luật Kế toán 2015 còn hiệu lực không?",
        retrieval_result=retrieval,
        top_k=5,
    )

    print(f"Context length: {len(augmented.context_text)} chars")
    print(f"Documents used: {len(augmented.documents)}")
    print(f"Token count (est): {augmented.token_count}")
    print(f"Strategy: {augmented.context_strategy}")
    print(f"\nContext preview:\n{augmented.context_text[:400]}...")

    print("\n✓ ContextAugmenter OK\n")


def test_generator():
    print("=" * 50)
    print("TEST 5: LegalAnswerGenerator (Phần 4)")
    print("=" * 50)

    from src.rag_pipeline.contracts import RetrievedDocument, RetrievalResult, AugmentedContext

    generator = LegalAnswerGenerator(generation_mode="extractive")

    docs = [
        RetrievedDocument(
            doc_id="doc_1",
            content="Luật Kế toán 2015 quy định về chế độ kế toán. Văn bản này còn hiệu lực. Điều 1 quy định phạm vi điều chỉnh.",
            metadata={"doc_type": "luat", "name": "Luật Kế toán 2015"},
            score=0.95,
            rank=1,
        ),
    ]

    context = AugmentedContext(
        original_question="Luật Kế toán 2015 còn hiệu lực không?",
        context_text="Câu hỏi: Luật Kế toán 2015 còn hiệu lực không?\n\n[luat] Luật Kế toán 2015\nLuật Kế toán 2015 quy định về chế độ kế toán. Văn bản này còn hiệu lực. Điều 1 quy định phạm vi điều chỉnh.",
        documents=docs,
        token_count=50,
        context_strategy="concat",
    )

    answer = generator.generate(context)

    print(f"Answer:\n{answer.answer_text}")
    print(f"\nConfidence: {answer.confidence}")
    print(f"Citations: {len(answer.citations)}")
    print(f"Reasoning steps: {answer.reasoning_steps}")
    print(f"Method: {answer.generation_method}")

    print("\n✓ LegalAnswerGenerator OK\n")


def test_full_pipeline():
    print("=" * 50)
    print("TEST 6: Full RAG Pipeline (Phần 2→3→4)")
    print("=" * 50)

    pipeline = RAGPipeline(search_engine=None, reasoner=None)

    request = RAGPipelineRequest(
        question="Luật Kế toán 2015 còn hiệu lực không?",
        top_k_retrieval=5,
        top_k_rerank=3,
    )

    response = pipeline.run(request)

    print(f"Answer:\n{response.answer}")
    print(f"\nConfidence: {response.confidence}")
    print(f"Sources: {len(response.sources)}")
    print(f"Reasoning steps: {response.reasoning}")
    print(f"Total latency: {response.latency_ms:.2f}ms")

    print("\n✓ Full RAG Pipeline OK\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("RAG PIPELINE TEST SUITE (Phần 2-3-4)")
    print("=" * 50 + "\n")

    test_mock_preprocessor()
    test_mock_postprocessor()
    test_retriever_without_search_engine()
    test_augmenter()
    test_generator()
    test_full_pipeline()

    print("=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
