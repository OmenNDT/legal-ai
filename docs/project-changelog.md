# LegalAI — Project Changelog

## [2026-05-08] RAG Pipeline Phần 2-3-4 Integration

### Added
- **[MAJOR]** `src/rag_pipeline/preprocessor.py` — `LoRAPreprocessor` loads trained LoRA checkpoint (`best_model.pt`) for real intent classification (20 classes) and NER (9 types), replacing keyword-based `MockPreprocessor`.
- **[MAJOR]** `src/rag_pipeline/query_expansion.py` — legal synonym expansion (`LEGAL_SYNONYMS`, `DOMAIN_MAP`) to boost retrieval recall.
- **[MAJOR]** `tests/test_rag_pipeline.py` + `tests/conftest.py` — pytest suite with 16 tests covering preprocessor, retriever, augmenter, generator, and full pipeline.

### Changed
- **[MAJOR]** `src/rag_pipeline/retrieval.py` — enabled vector search via `VectorStore` + `HybridSearch` (RRF fusion); integrated query expansion; fixed `use_vector=False` hardcode.
- **[MAJOR]** `src/rag_pipeline/augment.py` — cross-encoder reranker enabled by default; added configurable `reranker_model` and GPU support.
- **[MAJOR]** `src/rag_pipeline/generation.py` — added `"llm"` generation mode using `LLMClient` (Ollama/OpenAI-compatible) with structured Vietnamese output (`**Căn cứ pháp lý:**`, `**Phân tích:**`, `**Kết luận:**`, `**Khuyến nghị:**`); extractive remains fallback.
- **[MINOR]** `src/rag_pipeline/pipeline.py` — `RAGPipeline` now accepts `vector_store`, `hybrid_search`, `llm_client` parameters.
- **[MINOR]** `src/app.py` — `_startup()` initializes `VectorStore`, `HybridSearch`, and `LLMClient`; RAG endpoints use pipeline's real preprocessor.
- **[MINOR]** `src/common/config.py` — added `LORA_CHECKPOINT_PATH`.
- **[MINOR]** `requirements.txt` — added `peft>=0.7.0`, `pytest>=8.0.0`.

### Fixed
- **[MAJOR]** `best_model.pt` path corrected: moved from repo root to `data/models/lora_ke_toan/best_model.pt`.
- **[MINOR]** `.env` — switched LLM model from `qwen3.5:397b-cloud` to `qwen2.5:7b` (local Ollama) and updated `LLM_BASE_URL` / `CHROMA_HOST` to `localhost`.
- **[MINOR]** `requirements.txt` — bumped `chromadb>=1.5.0` for v2 API compatibility.
- **[MINOR]** `src/rag_pipeline/preprocessor.py` — `_build_filters()` now emits Vietnamese `doc_type` values (`luat`, `nghi_dinh`, `thong_tu`) to match ChromaDB metadata, fixing empty retrieval when NER inferred document type.
- **[MINOR]** `src/rag_pipeline/retrieval.py` — added pure vector search fallback when BM25 engine is unavailable; moved `_apply_filters()` after all retrieval paths.

### Removed
- **[MINOR]** `test_rag_pipeline.py` (root) — replaced by proper `tests/` pytest suite.
