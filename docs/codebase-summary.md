# LegalAI — Codebase Summary

## Project Statistics

| Metric | Value |
|--------|-------|
| Total LOC (Python) | ~4,595 |
| Source files (`src/`) | 18 (.py) |
| Script files (`scripts/`) | 9 (.py) |
| Config constants | ~110 |
| PhoBERT heads | 3 (intent, NER, scorer) |
| LoRA checkpoint | `data/models/lora_ke_toan/best_model.pt` |
| Knowledge graph nodes | 20,382 |
| Knowledge graph edges | 57,047 |
| Search index docs | 599 |
| Search index terms | 5,454 |
| Test files | 0 |

## Directory Structure

```
legal-ai/
├── data/
│   ├── raw/                         # Downloaded source data
│   │   ├── yuiTC_sample.json        # 89,261 QA pairs
│   │   └── uts_vlc_processed.json   # 600 legal documents
│   ├── processed/                   # Preprocessed / built artifacts
│   │   ├── search_index.json        # Serialized inverted index
│   │   ├── knowledge_graph.gpickle  # NetworkX graph
│   │   ├── legal_docs.json          # Cleaned legal documents
│   │   ├── qa_train.json            # Train split (80%)
│   │   ├── qa_val.json              # Val split (10%)
│   │   ├── qa_test.json             # Test split (10%)
│   │   ├── luat_ke_toan_2025_structured.json  # Parsed law hierarchy
│   │   ├── qa_ke_toan_train.json    # Generated QA v1
│   │   └── qa_ke_toan_train_v2.json # Generated QA v2 (expanded)
│   └── models/                      # Trained model weights
│       ├── intent_classifier/       # (unused — LoRA replaces this)
│       ├── ner_tagger/              # (unused — LoRA replaces this)
│       ├── sentence_scorer/         # (unused — TF-IDF fallback)
│       └── lora_ke_toan/
│           └── best_model.pt          # LoRA multi-task checkpoint
├── src/
│   ├── app.py                       # FastAPI backend (all endpoints)
│   ├── ui.py                        # Streamlit frontend (4 tabs)
│   ├── common/
│   │   ├── config.py                # Constants, paths, hyperparameters
│   │   ├── data_loader.py           # QA and document data loading
│   │   └── text_processor.py        # VnCoreNLP segmentation, NER label builder
│   ├── search/
│   │   ├── inverted_index.py        # Positional inverted index (166 LOC)
│   │   ├── bm25.py                  # Okapi BM25 scoring (96 LOC)
│   │   ├── trie.py                  # Trie + Levenshtein fuzzy search (109 LOC)
│   │   └── search_engine.py         # Unified LegalSearchEngine facade (67 LOC)
│   ├── summarizer/
│   │   ├── sentence_scorer.py         # PhoBERT Head 3: sentence importance (111 LOC)
│   │   ├── textrank.py              # PageRank + position scoring (157 LOC)
│   │   └── summarizer.py            # Extractive summarizer orchestrator (132 LOC)
│   ├── chatbot/
│   │   ├── intent_classifier.py       # PhoBERT Head 1: 20-intent classification (79 LOC)
│   │   ├── ner_tagger.py            # PhoBERT Head 2: BIO NER tagging (123 LOC)
│   │   └── pipeline.py              # LegalChatbot orchestrator (188 LOC)
│   ├── knowledge/
│   │   ├── entity_extractor.py      # Regex entity/relation extraction (225 LOC)
│   │   ├── graph_builder.py         # NetworkX DiGraph builder (284 LOC)
│   │   ├── reasoner.py              # High-level KG queries (168 LOC)
│   │   └── visualizer.py            # Pyvis HTML visualization (143 LOC)
│   └── __init__.py files            # Package markers
├── scripts/
│   ├── download_data.py             # Fetch raw data from GitHub (55 LOC)
│   ├── preprocess.py                  # Segment, clean, split train/val/test (154 LOC)
│   ├── build_index.py               # Build search index + KG from docs (98 LOC)
│   ├── build_graph.py               # Build KG from legal docs + QA data (130 LOC)
│   ├── parse_luat_ke_toan.py        # Parse Luat Ke toan 2025 hierarchy (221 LOC)
│   ├── generate_qa_dataset.py         # Generate QA pairs v1 from parsed law (214 LOC)
│   ├── generate_qa_dataset_v2.py      # Expanded QA dataset v2 with paraphrasing (347 LOC)
│   ├── train_lora_phobert.py          # LoRA multi-task training (intent + NER) (272 LOC)
│   └── inference_lora.py              # LoRA inference + interactive CLI (234 LOC)
├── tests/                           # Empty — no tests yet
├── notebooks/                       # Empty — no notebooks yet
├── docs/                            # Documentation
├── requirements.txt
└── README.md
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
- **LegalChatbot**: Pipeline orchestrator with INTENT_ROUTING table; lazy model loading; supports loading LoRA checkpoint

### Knowledge Graph Module (AI)
- **LegalEntityExtractor**: Regex patterns for 9 entity types (LUAT, THONG_TU, NGHI_DINH, DIEU, KHOAN, DIEM, CO_QUAN, KHAISUAT, NGAY_THANG)
- **LegalRelationExtractor**: Regex patterns for 7 relation types (DUA_TREN, THAM_CHIEU, HET_HIEU_LUC, THAY_THE, SUA_DOI_BO_SUNG, HUONG_DAN, CHUA)
- **LegalKnowledgeGraph**: NetworkX DiGraph with stats, validity check, amendment chain, BFS related docs, shortest path
- **LegalReasoner**: High-level query API returning LegalAnswer with confidence, evidence, reasoning_steps
- **Visualizer**: Pyvis interactive HTML export with node/edge coloring by type

### LoRA Training Pipeline
- **parse_luat_ke_toan.py**: Parses raw law text into structured JSON (Chuong -> Dieu -> Khoan -> Diem hierarchy)
- **generate_qa_dataset_v2.py**: Expanded QA generation with 10 intent categories, up to 14 templates per intent, synonym paraphrasing
- **train_lora_phobert.py**: Multi-task LoRA fine-tuning (r=16, alpha=32, dropout=0.1) on intent + NER; AdamW, linear warmup, gradient clipping
- **inference_lora.py**: Loads checkpoint, runs demo/interactive/single-question inference with top-3 intent predictions and entity extraction
