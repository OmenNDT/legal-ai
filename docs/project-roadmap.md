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
| Week 1: Foundation | In Progress | Week 1 | Week 1 |
| Week 2: Training | Not Started | Week 2 | Week 2 |
| Week 3: Integration | Not Started | Week 3 | Week 3 |
| Week 4: Polish | Not Started | Week 4 | Week 4 |

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| PhoBERT training too slow on CPU | Blocks Week 2 | Use Google Colab GPU or reduce epochs |
| Insufficient labeled NER data | Poor NER accuracy | Use regex-extracted labels as silver standard |
| VnCoreNLP segmentation errors | Affects all PhoBERT inputs | Add segmentation validation step |
| Knowledge graph sparse coverage | Low reasoning quality | Augment with QA-derived entities |
| Large document summarization OOM | API failures | Truncate documents to MAX_SEQ_LENGTH |

## Deliverables Summary

1. **Search**: Custom InvertedIndex + BM25 + Trie + Levenshtein with tests and benchmarks
2. **Summarizer**: PhoBERT + TextRank extractive summarizer with ROUGE evaluation
3. **Chatbot**: Trained intent classifier (20 classes) + NER tagger (9 types) + routing pipeline
4. **Knowledge Graph**: NetworkX graph with 20K+ nodes, reasoning queries, Pyvis visualization
5. **API**: FastAPI with all endpoints, error handling, integration tests
6. **UI**: Streamlit with 4 tabs, formatted results, API health check
7. **Documentation**: Architecture docs, deployment guide, code standards