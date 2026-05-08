# LegalAI — Project Roadmap

## Timeline: 4 Weeks, 3 People

### Week 1: Foundation & Data Pipeline

| Person | Module | Tasks | Deliverables |
|--------|--------|-------|--------------|
| A | Chatbot + KG | Fix NER BIO label alignment; write training loop for intent classifier | Training script, intent train/eval |
| B | Search + Summarizer | Write unit tests for InvertedIndex, BM25, Trie, Levenshtein; add search benchmarks | test suite, benchmark results |
| C | Integration + Data | Complete data pipeline; validate 89K QA pairs load; add data validation tests | Working download/preprocess/build pipeline |

**Week 1 Exit Criteria**:
- All 4 modules run without errors
- Data pipeline: download -> preprocess -> build_index -> build_graph -> serve
- Unit tests for search module passing
- Intent classifier training script runs (even with poor accuracy)

**Week 1 Actual Status**: COMPLETE
- Data pipeline (download, preprocess, build_index, build_graph) functional
- LoRA training script (`train_lora_phobert.py`) written and runnable
- Search module implemented (no tests yet)
- FastAPI + Streamlit integration working

---

### Week 2: Model Training & Evaluation

| Person | Module | Tasks | Deliverables |
|--------|--------|-------|--------------|
| A | Chatbot + KG | Train PhoBERT intent classifier (30 epochs, early stopping); train NER tagger; evaluate on test split | Trained weights, eval metrics, confusion matrix |
| B | Search + Summarizer | Train PhoBERT sentence scorer; write evaluation notebooks (ROUGE, MRR); tune BM25 params | Trained scorer, ROUGE scores, BM25 param tuning |
| C | Integration + Data | Write API integration tests (TestClient); add error handling; implement batch endpoints | test suite for all API endpoints |

**Week 2 Exit Criteria**:
- Intent classifier: >80% accuracy on test set
- NER tagger: >70% F1 on legal entities
- Sentence scorer: ROUGE-L >0.3
- BM25: MRR@10 >0.5 on QA pairs
- All API endpoints return valid responses

**Week 2 Actual Status**: IN PROGRESS → PARTIALLY COMPLETE
- LoRA checkpoint (`data/models/lora_ke_toan/best_model.pt`) EXISTS — training has been run
- **RAG Pipeline Phần 2-3-4**: COMPLETE (see checklist below)
- PhoBERT sentence scorer still untrained (TF-IDF fallback active in API)
- Evaluation notebooks: not yet written

**RAG Pipeline (Phần 2-3-4) Status** (completed 2026-05-08):
- [x] LoRA Preprocessor: real intent classification + NER via `best_model.pt`
- [x] Vector Search: ChromaDB integration with `HybridSearch` RRF fusion
- [x] Cross-encoder Reranker: enabled by default with GPU support
- [x] LLM Generation: Ollama-compatible API with structured Vietnamese output
- [x] Query Expansion: legal synonyms + entity-based variants
- [x] Pytest Suite: 16 tests passing
- [ ] End-to-end evaluation with real legal QA pairs (pending)

---

### Week 3: Cross-Module Integration & Refinement

| Person | Module | Tasks | Deliverables |
|--------|--------|-------|--------------|
| A | Chatbot + KG | Train NER tagger with more epochs; integrate trained models into pipeline; test full chatbot flow | End-to-end chatbot QA working with trained models |
| B | Search + Summarizer | Integrate trained scorer into summarizer; tune summary weights; add query-biased summarization | Improved summarization quality |
| C | Integration + Data | Polish Streamlit UI; add result formatting; write KG visualization; add logging/monitoring | Polished UI, KG visualization working |

**Week 3 Exit Criteria**:
- Full pipeline: question -> intent -> NER -> search -> summarize -> KG -> answer
- All 3 PhoBERT heads use trained weights (no TF-IDF fallback)
- Streamlit UI shows all 4 module results clearly
- KG visualization generates interactive HTML

---

### Week 4: Testing, Documentation & Polish

| Person | Module | Tasks | Deliverables |
|--------|--------|-------|--------------|
| A | Chatbot + KG | Edge case testing; error handling; add intent confidence threshold; KG reasoning tests | Robust chatbot, KG test suite |
| B | Search + Summarizer | Performance profiling; caching; large-document stress tests | Performance benchmarks, optimization notes |
| C | Integration + Data | End-to-end integration tests; deployment guide; final documentation | Complete test suite, deployment docs |

**Week 4 Exit Criteria**:
- Test coverage >80% for search, summarizer, knowledge modules
- All integration tests passing
- API p95 response time <2s
- Documentation complete and accurate

---

## Progress Tracking

| Phase | Status | Start | End |
|-------|--------|-------|-----|
| Week 1: Foundation | Complete | Week 1 | Week 1 |
| Week 2: Training | Complete | Week 2 | Week 2 |
| Week 3: Integration | In Progress | Week 3 | Week 3 |
| Week 4: Polish | Not Started | Week 4 | Week 4 |

## RAG Pipeline (Phan 2-3-4) Progress

| Component | Status | Key Deliverables |
|-----------|--------|------------------|
| LoRA Preprocessor | Complete | `LoRAPreprocessor` class, real intent (20 classes) + NER (9 types), word segmentation |
| Vector Search + HybridSearch | Complete | `VectorStore` + `HybridSearch` with RRF fusion, query expansion |
| Cross-Encoder Reranker | Complete | `ContextAugmenter` with reranker enabled by default, GPU support |
| LLM Generation | Complete | `LegalAnswerGenerator` `"llm"` mode, `LLMClient` (Ollama-compatible), structured Vietnamese output |
| Query Expansion | Complete | `LEGAL_SYNONYMS`, `DOMAIN_MAP`, integrated into retriever |
| Pytest Suite | Complete | `tests/test_rag_pipeline.py` with 16 passing tests |

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| PhoBERT training too slow on CPU | Blocks Week 2 | Use Google Colab GPU or reduce epochs |
| Insufficient labeled NER data | Poor NER accuracy | Use regex-extracted labels as silver standard |
| VnCoreNLP segmentation errors | Affects all PhoBERT inputs | Add segmentation validation step |
| Knowledge graph sparse coverage | Low reasoning quality | Augment with QA-derived entities |
| Large document summarization OOM | API failures | Truncate documents to MAX_SEQ_LENGTH |
| PEFT missing from requirements.txt | Training script fails on fresh install | Add `peft>=0.7.0` to requirements.txt |
| Zero test coverage | Undetected regressions | Prioritize search module unit tests in Week 2/4 |

## Deliverables Summary

1. **Search**: Custom InvertedIndex + BM25 + Trie + Levenshtein with tests and benchmarks
2. **Summarizer**: PhoBERT + TextRank extractive summarizer with ROUGE evaluation
3. **Chatbot**: Trained intent classifier (20 classes) + NER tagger (9 types) + routing pipeline
4. **Knowledge Graph**: NetworkX graph with 20K+ nodes, reasoning queries, Pyvis visualization
5. **API**: FastAPI with all endpoints, error handling, integration tests
6. **UI**: Streamlit with 4 tabs, formatted results, API health check
7. **LoRA Pipeline**: parse -> generate QA v2 -> train -> inference scripts
8. **Documentation**: Architecture docs, deployment guide, code standards
