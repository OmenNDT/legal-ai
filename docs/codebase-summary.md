# LegalAI — Codebase Summary

## Project Statistics

| Metric | Value |
|--------|-------|
| Total LOC (Python) | ~3,342 |
| Source files | 26 (.py) |
| Script files | 4 (.py) |
| Config constants | ~110 |
| PhoBERT heads | 3 (intent, NER, scorer) |
| Knowledge graph nodes | 20,382 |
| Knowledge graph edges | 57,047 |
| Search index docs | 599 |
| Search index terms | 5,454 |

## Directory Structure

```
legal-ai/
├── data/
│   ├── raw/                    # Downloaded source data
│   │   ├── yuiTC_sample.json  # 89,261 QA pairs
│   │   └── uts_vlc_processed.json  # 600 legal documents
│   ├── processed/             # Preprocessed data (built by scripts)
│   │   ├── search_index.json  # Serialized inverted index
│   │   ├── knowledge_graph.gpickle  # NetworkX graph
│   │   ├── legal_docs.json   # Cleaned legal documents
│   │   ├── qa_train.json     # Train split (80%)
│   │   ├── qa_val.json       # Val split (10%)
│   │   └── qa_test.json      # Test split (10%)
│   └── models/                # Trained model weights (empty until training)
│       ├── intent_classifier/
│       ├── ner_tagger/
│       └── sentence_scorer/
├── src/
│   ├── app.py                 # FastAPI backend (all endpoints)
│   ├── ui.py                  # Streamlit frontend (4 tabs)
│   ├── common/
│   │   ├── config.py          # Constants, paths, hyperparameters
│   │   ├── data_loader.py     # QA and document data loading
│   │   └── text_processor.py  # VnCoreNLP segmentation, NER label generation
│   ├── search/
│   │   ├── inverted_index.py  # InvertedIndex with positional posting lists
│   │   ├── bm25.py            # BM25 ranking (Okapi BM25)
│   │   ├── trie.py             # Trie + Levenshtein fuzzy search
│   │   └── search_engine.py   # Unified LegalSearchEngine facade
│   ├── summarizer/
│   │   ├── sentence_scorer.py # PhoBERT Head 3: sentence importance scoring
│   │   ├── textrank.py        # TextRank PageRank + position scoring + selection
│   │   └── summarizer.py      # LegalSummarizer orchestrating all scoring
│   ├── chatbot/
│   │   ├── intent_classifier.py # PhoBERT Head 1: 20-intent classification
│   │   ├── ner_tagger.py       # PhoBERT Head 2: 9-entity BIO tagging
│   │   └── pipeline.py        # LegalChatbot orchestrator (INTENT_ROUTING)
│   ├── knowledge/
│   │   ├── entity_extractor.py # Regex entity + relation extraction
│   │   ├── graph_builder.py   # LegalKnowledgeGraph (NetworkX DiGraph)
│   │   ├── reasoner.py        # LegalReasoner (validity, amendments, paths)
│   │   └── visualizer.py     # Pyvis HTML visualization
│   └── __init__.py files      # Package markers
├── scripts/
│   ├── download_data.py       # Download data from GitHub
│   ├── preprocess.py          # Segment text, build train/val/test splits
│   ├── build_index.py         # Build search index + knowledge graph
│   └── build_graph.py         # Build KG from legal docs + QA data
├── tests/                     # Empty — no tests yet
├── notebooks/                 # Empty — no notebooks yet
├── docs/                      # Documentation
└── requirements.txt           # Python dependencies
```

## Module Details

### Search Module (Algorithm Design)
- **InvertedIndex**: Hash map + sorted posting lists with positional info; O(k) term lookup, O(min posting) AND search
- **BM25**: Okapi BM25 scoring with IDF precomputation; configurable k1=1.5, b=0.75
- **Trie**: Prefix tree for autocomplete with frequency-based ranking; O(m+k) prefix completion
- **Levenshtein**: DP edit distance with early termination; fuzzy search over vocabulary
- **LegalSearchEngine**: Facade combining all 4 components; `search()`, `boolean_search()`, `autocomplete()`, `fuzzy()`, `explain()`

### Summarizer Module (Python ML)
- **PhoBERTSentenceScorer**: PhoBERT -> mean pooling -> [sent, doc, sent*doc] -> Linear(2304,1); scores sentence importance
- **TextRank**: Cosine similarity graph -> PageRank; configurable damping=0.85, max_iter=100, tolerance=1e-5
- **LegalSummarizer**: Combined scoring = 0.4*relevance + 0.4*centrality + 0.2*position; TF-IDF fallback when model unavailable

### Chatbot Module (NLP)
- **PhoBERTIntentClassifier**: PhoBERT -> [CLS] -> Dropout -> Linear(768, 20); 20 intent classes
- **PhoBERTNERTagger**: PhoBERT -> Linear(768, 19); 9 entity types with BIO tagging
- **LegalChatbot**: Pipeline orchestrator with INTENT_ROUTING table; lazy model loading

### Knowledge Graph Module (AI)
- **LegalEntityExtractor**: Regex patterns for 7 entity types (THONG_TU, NGHI_DINH, LUAT, DIEU, KHOAN, DIEM, CO_QUAN, NGAY_THANG)
- **LegalRelationExtractor**: Regex patterns for 7 relation types (DUA_TREN, THAM_CHIEU, HET_HIEU_LUC, THAY_THE, SUA_DOI_BO_SUNG, HUONG_DAN, CHUA)
- **LegalKnowledgeGraph**: NetworkX DiGraph with stats, validity check, amendment chain, BFS related docs, shortest path
- **LegalReasoner**: High-level query API returning LegalAnswer with confidence, evidence, reasoning_steps
- **Visualizer**: Pyvis interactive HTML export with node/edge coloring by type