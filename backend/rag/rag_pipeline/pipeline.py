"""RAG Pipeline Orchestrator: kết nối Phần 2 → Phần 3 → Phần 4.

Đây là trái tim của module RAG, điều phối luồng dữ liệu giữa
Retrieval → Augment → Generation.
"""

from typing import Optional

from backend.rag_pipeline.contracts import (
    ProcessedQuestion,
    RAGPipelineRequest,
    RAGPipelineResponse,
    Citation,
)
from backend.rag_pipeline.retrieval import LegalRetriever
from backend.rag_pipeline.augment import ContextAugmenter
from backend.rag_pipeline.generation import LegalAnswerGenerator
from backend.search.search_engine import LegalSearchEngine
from backend.knowledge.reasoner import LegalReasoner


class RAGPipeline:
    """Pipeline RAG hoàn chỉnh: Retrieval → Augment → Generation.

    Usage:
        pipeline = RAGPipeline(search_engine=search_engine)
        response = pipeline.run(RAGPipelineRequest(question="..."))
    """

    def __init__(
        self,
        search_engine: Optional[LegalSearchEngine] = None,
        reasoner: Optional[LegalReasoner] = None,
        retriever: Optional[LegalRetriever] = None,
        augmenter: Optional[ContextAugmenter] = None,
        generator: Optional[LegalAnswerGenerator] = None,
        vector_store=None,
        hybrid_search=None,
        llm_client=None,
    ):
        # Khởi tạo các thành phần với defaults
        self.retriever = retriever or LegalRetriever(
            search_engine=search_engine,
            use_vector=True,
            vector_store=vector_store,
            hybrid_search=hybrid_search,
        )
        self.augmenter = augmenter or ContextAugmenter(
            use_reranker=True,
        )
        self.generator = generator or LegalAnswerGenerator(
            generation_mode="llm" if llm_client else "extractive",
            llm_client=llm_client,
        )
        self.reasoner = reasoner

        # Preprocessor: real LoRA if checkpoint exists, else mock
        from backend.rag_pipeline.preprocessor import LoRAPreprocessor
        from backend.common.config import LORA_CHECKPOINT_PATH

        if LORA_CHECKPOINT_PATH.exists():
            self.preprocessor = LoRAPreprocessor(checkpoint_path=str(LORA_CHECKPOINT_PATH))
        else:
            from backend.rag_pipeline.mock_adapters import MockPreprocessor

            self.preprocessor = MockPreprocessor()

    def run(self, request: RAGPipelineRequest) -> RAGPipelineResponse:
        """Chạy toàn bộ pipeline 2→3→4.

        Args:
            request: RAGPipelineRequest với câu hỏi và config

        Returns:
            RAGPipelineResponse với câu trả lời đầy đủ
        """
        import time
        total_start = time.time()

        # ── Bước 0: Tạo ProcessedQuestion từ raw question ──
        processed = self.preprocessor.process(request.question)

        # ── Phần 2: RETRIEVAL ──
        retrieval_result = self.retriever.retrieve(
            question=processed,
            top_k=request.top_k_retrieval,
            filters=processed.filters,
        )

        # ── Phần 3: AUGMENT ──
        if self.reasoner is not None:
            augmented = self.augmenter.augment_with_kg(
                question=request.question,
                retrieval_result=retrieval_result,
                reasoner=self.reasoner,
                top_k=request.top_k_rerank,
            )
        else:
            augmented = self.augmenter.augment(
                question=request.question,
                retrieval_result=retrieval_result,
                top_k=request.top_k_rerank,
            )

        # ── Phần 4: GENERATION ──
        generated = self.generator.generate(
            context=augmented,
            max_length=request.max_context_tokens,
        )

        total_latency = (time.time() - total_start) * 1000

        return RAGPipelineResponse(
            answer=generated.answer_text,
            confidence=generated.confidence,
            sources=generated.citations,
            reasoning=generated.reasoning_steps,
            retrieval_info=retrieval_result,
            latency_ms=total_latency,
        )

    def run_from_processed(
        self,
        processed: ProcessedQuestion,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5,
        max_context_tokens: int = 1024,
    ) -> RAGPipelineResponse:
        """Chạy pipeline từ ProcessedQuestion (đã qua Phần 1).

        Dùng khi Phần 1 đã xử lý xong và truyền vào.
        """
        request = RAGPipelineRequest(
            question=processed.raw_text,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
            max_context_tokens=max_context_tokens,
        )
        # Override processed question trong retriever
        retrieval_result = self.retriever.retrieve(
            question=processed,
            top_k=top_k_retrieval,
            filters=processed.filters,
        )

        if self.reasoner is not None:
            augmented = self.augmenter.augment_with_kg(
                question=processed.raw_text,
                retrieval_result=retrieval_result,
                reasoner=self.reasoner,
                top_k=top_k_rerank,
            )
        else:
            augmented = self.augmenter.augment(
                question=processed.raw_text,
                retrieval_result=retrieval_result,
                top_k=top_k_rerank,
            )

        generated = self.generator.generate(context=augmented)

        return RAGPipelineResponse(
            answer=generated.answer_text,
            confidence=generated.confidence,
            sources=generated.citations,
            reasoning=generated.reasoning_steps,
            retrieval_info=retrieval_result,
            latency_ms=retrieval_result.latency_ms + generated.latency_ms,
        )
