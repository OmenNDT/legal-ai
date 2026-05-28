"""RAG Pipeline module for LegalAI.

Phần 2-3-4 trong kiến trúc RAG:
- Phần 2: Retrieval (Truy hồi tài liệu)
- Phần 3: Augment (Bổ sung ngữ cảnh)
- Phần 4: Generation (Sinh câu trả lời)
"""

from backend.rag_pipeline.contracts import (
    ProcessedQuestion,
    RetrievalResult,
    RetrievedDocument,
    AugmentedContext,
    GeneratedAnswer,
    RAGPipelineRequest,
    RAGPipelineResponse,
)
from backend.rag_pipeline.retrieval import LegalRetriever
from backend.rag_pipeline.augment import ContextAugmenter
from backend.rag_pipeline.generation import LegalAnswerGenerator
from backend.rag_pipeline.pipeline import RAGPipeline
from backend.rag_pipeline.mock_adapters import MockPreprocessor, MockPostprocessor
from backend.rag_pipeline.preprocessor import LoRAPreprocessor

__all__ = [
    "ProcessedQuestion",
    "RetrievalResult",
    "RetrievedDocument",
    "AugmentedContext",
    "GeneratedAnswer",
    "RAGPipelineRequest",
    "RAGPipelineResponse",
    "LegalRetriever",
    "ContextAugmenter",
    "LegalAnswerGenerator",
    "RAGPipeline",
    "MockPreprocessor",
    "MockPostprocessor",
    "LoRAPreprocessor",
]
