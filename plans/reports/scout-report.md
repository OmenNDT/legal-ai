# LegalAI Codebase Scout Report

**Date:** 2026-04-27
**Project:** LegalAI — Vietnamese Legal AI Assistant
**Root:** `/home/sontn/Projects/legal-ai`

---

## 1. Directory Structure with File Counts and LOC

| Directory | Files | LOC | Description |
|-----------|-------|-----|-------------|
| `src/` | 18 `.py` files | 2,879 | Core source modules |
| `scripts/` | 9 `.py` files | 1,716 | Data pipeline, training, and inference scripts |
| `docs/` | 7 `.md` files | 781 | Architecture, deployment, standards, roadmap |
| **Total Python** | **27** | **4,595** | |
| `data/raw/` | 3 files | — | Source data (`yuiTC_sample.json`, `uts_vlc_processed.json`, `luat_ke_toan_2025.txt`) |
| `data/processed/` | 7 files | — | Built artifacts (indices, graphs, QA splits) |
| `data/models/` | 1 checkpoint | — | `lora_ke_toan/best_model.pt` |

### Full File Tree

```
legal-ai/
├── src/
│   ├── app.py                      # FastAPI backend (196 LOC)
│   ├── ui.py                       # Streamlit frontend (184 LOC)
│   ├── __init__.py                 # Package marker
│   ├── common/
│   │   ├── config.py               # Constants, paths, hyperparameters (110 LOC)
│   │   ├── data_loader.py          # QA/document loading utilities (95 LOC)
│   │   └── text_processor.py       # VnCoreNLP segmentation, NER label builder (140 LOC)
│   ├── search/
│   │   ├── __init__.py
│   │   ├── inverted_index.py       # Positional inverted index (166 LOC)
│   │   ├── bm25.py                 # Okapi BM25 scoring (96 LOC)
│   │   ├── trie.py                 # Trie + Levenshtein fuzzy search (109 LOC)
│   │   └── search_engine.py        # Unified search facade (67 LOC)
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py    # PhoBERT Head 1: 20-intent classifier (79 LOC)
│   │   ├── ner_tagger.py           # PhoBERT Head 2: BIO NER tagger (123 LOC)
│   │   └── pipeline.py             # LegalChatbot orchestrator (188 LOC)
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── entity_extractor.py     # Regex entity/relation extraction (225 LOC)
│   │   ├── graph_builder.py        # NetworkX DiGraph builder (284 LOC)
│   │   ├── reasoner.py             # High-level KG queries (168 LOC)
│   │   └── visualizer.py           # Pyvis HTML visualization (143 LOC)
│   └── summarizer/
│       ├── __init__.py
│       ├── sentence_scorer.py      # PhoBERT Head 3: sentence importance (111 LOC)
│       ├── textrank.py             # PageRank + position scoring (157 LOC)
│       └── summarizer.py           # Extractive summarizer orchestrator (132 LOC)
├── scripts/
│   ├── download_data.py            # Fetch raw data from GitHub (55 LOC)
│   ├── preprocess.py                 # Segment, clean, split train/val/test (154 LOC)
│   ├── build_index.py                # Build search index + KG from docs (98 LOC)
│   ├── build_graph.py                # Build KG from legal docs + QA data (130 LOC)
│   ├── parse_luat_ke_toan.py         # Parse Luat Ke toan 2025 hierarchy (221 LOC)
│   ├── generate_qa_dataset.py        # Generate QA pairs v1 from parsed law (214 LOC)
│   ├── generate_qa_dataset_v2.py   # Expanded QA dataset v2 with paraphrasing (347 LOC)
│   ├── train_lora_phobert.py        # LoRA multi-task training (intent + NER) (272 LOC)
│   └── inference_lora.py            # LoRA inference + interactive CLI (234 LOC)
├── docs/
│   ├── architecture.md
│   ├── codebase-summary.md
│   ├── code-standards.md
│   ├── deployment-guide.md
│   ├── project-overview-pdr.md
│   ├── project-roadmap.md
│   └── system-architecture.md
├── requirements.txt
└── data/
    ├── raw/
    ├── processed/
    └── models/
```

---

## 2. Module Descriptions

### search/ (Algorithm Design Course)
Custom legal document search with no external search libraries.
- **InvertedIndex**: Hash map of term -> positional posting lists. Supports AND, OR, NOT, and phrase search.
- **BM25**: Okapi BM25 ranking with configurable `k1=1.5`, `b=0.75`. Includes per-term `explain()` breakdown.
- **Trie**: Prefix tree for autocomplete with frequency-based ranking.
- **Levenshtein**: DP edit distance with early termination for fuzzy vocabulary search.
- **LegalSearchEngine**: Facade combining all four components.

### chatbot/ (NLP Course)
PhoBERT-based question understanding and orchestration.
- **PhoBERTIntentClassifier**: 20-class intent classifier (PhoBERT -> [CLS] -> Linear).
- **PhoBERTNERTagger**: 9-entity BIO tagger (PhoBERT -> token-level Linear).
- **LegalChatbot**: Pipeline orchestrator that routes intents to search/summarizer/KG modules.

### knowledge/ (AI Course)
NetworkX-based legal knowledge graph.
- **LegalEntityExtractor**: Regex extraction of 9 entity types.
- **LegalRelationExtractor**: Regex extraction of 7 relation types.
- **LegalKnowledgeGraph**: DiGraph with validity checks, amendment chains, BFS related docs, shortest paths.
- **LegalReasoner**: High-level query API returning structured `LegalAnswer` with evidence and reasoning steps.
- **Visualizer**: Pyvis interactive HTML export with type-based node/edge coloring.

### summarizer/ (Python ML Course)
Extractive summarization avoiding abstractive hallucination.
- **PhoBERTSentenceScorer**: Sentence-document relevance scoring via concatenated embeddings.
- **TextRank**: Cosine similarity graph -> PageRank (damping 0.85).
- **LegalSummarizer**: Combines relevance (40%), centrality (40%), position (20%) with TF-IDF fallback.

### common/
Shared infrastructure.
- **config.py**: All paths, model names, hyperparameters, label lists.
- **data_loader.py**: `load_qa_data()`, `load_legal_documents()`.
- **text_processor.py**: VnCoreNLP word segmentation, sentence splitting, legal reference extraction, BIO label builder.

---

## 3. Key Classes and Public APIs

### Search
| Class | Key Methods |
|-------|-------------|
| `InvertedIndex` | `add_document()`, `search(term)`, `search_and/or/not/phrase()`, `save()`, `load()` |
| `BM25` | `search(query, top_k)`, `explain(query, doc_id)` |
| `Trie` | `insert(word, freq)`, `autocomplete(prefix, max_results)` |
| `LegalSearchEngine` | `search()`, `boolean_search()`, `autocomplete()`, `fuzzy()`, `explain()` |

### Chatbot
| Class | Key Methods |
|-------|-------------|
| `PhoBERTIntentClassifier` | `forward()`, `predict(text, tokenizer)` |
| `PhoBERTNERTagger` | `forward()`, `predict(text, tokenizer)` |
| `LegalChatbot` | `ask(question, top_k_docs, top_k_sentences)` -> `ChatResponse` |

### Knowledge Graph
| Class | Key Methods |
|-------|-------------|
| `LegalKnowledgeGraph` | `add_document()`, `is_valid()`, `amendment_chain()`, `related_docs()`, `reasoning_path()`, `save()`, `load()` |
| `LegalReasoner` | `check_validity()`, `trace_amendments()`, `find_related()`, `find_reasoning_path()` |
| `LegalEntityExtractor` | `extract(text)` -> list[`LegalEntity`] |
| `LegalRelationExtractor` | `extract(text, doc_entity)` -> list[`LegalRelation`] |

### Summarizer
| Class | Key Methods |
|-------|-------------|
| `PhoBERTSentenceScorer` | `forward()`, `score_sentences(sentences, document, tokenizer)` |
| `LegalSummarizer` | `summarize(document, query, top_k, use_model)` |

### FastAPI Endpoints (`src/app.py`)
- `GET /` — platform status
- `GET /health` — health check
- `POST /search` — BM25 ranked search
- `POST /search/autocomplete` — Trie prefix suggestions
- `POST /search/explain` — BM25 score breakdown
- `POST /summarize` — extractive summarization
- `POST /knowledge/query` — KG reasoning (`validity`, `amendments`, `related`, `path`)
- `POST /knowledge/stats` — graph statistics
- `POST /knowledge/visualize` — Pyvis HTML export
- `POST /chat` — full chatbot pipeline

---

## 4. Data Pipeline Flow

```
1. download_data.py
   → Downloads yuiTC_sample.json (89,261 QA pairs)
   → Downloads uts_vlc_processed.json (600 legal docs)

2. preprocess.py
   → Word segmentation (VnCoreNLP)
   → Legal reference extraction
   → NER BIO label generation
   → Train/val/test split (80/10/10)
   → Outputs: qa_train.json, qa_val.json, qa_test.json, legal_docs.json

3. build_index.py
   → Builds InvertedIndex + BM25 + Trie
   → Builds Knowledge Graph (NetworkX DiGraph)
   → Outputs: search_index.json, knowledge_graph.gpickle

4. (Optional) parse_luat_ke_toan.py
   → Parses luat_ke_toan_2025.txt into structured JSON
   → Output: luat_ke_toan_2025_structured.json

5. (Optional) generate_qa_dataset.py / v2
   → Generates QA pairs from structured law
   → Outputs: qa_ke_toan_train.json, qa_ke_toan_train_v2.json

6. (Optional) train_lora_phobert.py
   → Fine-tunes PhoBERT with LoRA (r=16, alpha=32) on intent + NER
   → Output: data/models/lora_ke_toan/best_model.pt

7. Serve
   → uvicorn src.app:app --port 8000
   → streamlit run src/ui.py
```

---

## 5. Configuration and Constants

**File:** `src/common/config.py`

### Paths
- `BASE_DIR`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `MODELS_DIR`
- `QA_DATA_PATH`, `LEGAL_DATA_PATH`
- `INTENT_MODEL_DIR`, `NER_MODEL_DIR`, `SCORER_MODEL_DIR`
- `SEARCH_INDEX_PATH`, `KG_PATH`

### Model Settings
- `PHOBERT_MODEL = "vinai/phobert-base"`
- `MAX_SEQ_LENGTH = 256`

### Search
- `BM25_K1 = 1.5`, `BM25_B = 0.75`
- `SEARCH_TOP_K = 10`
- `FUZZY_MAX_DIST = 2`

### Summarizer
- `SUMMARY_WEIGHTS = {"relevance": 0.4, "centrality": 0.4, "position": 0.2}`
- `TEXTRANK_DAMPING = 0.85`, `TEXTRANK_MAX_ITER = 100`, `TEXTRANK_TOLERANCE = 1e-5`

### Training
- `TRAIN_BATCH_SIZE = 32`, `SCORER_BATCH_SIZE = 16`
- `LEARNING_RATE = 1e-5`, `SCORER_LEARNING_RATE = 2e-5`
- `NUM_EPOCHS = 30`, `EARLY_STOPPING_PATIENCE = 5`, `FP16 = True`

### Labels
- **Intent:** 21 classes (`hoi_dieu_khoan`, `hoi_luat_moi`, `tra_cuu_thong_tu`, ... `hoi_tong_hop`)
- **Entity:** 9 types (`LUAT`, `THONG_TU`, `NGHI_DINH`, `DIEU`, `KHOAN`, `DIEM`, `CO_QUAN`, `KHAISUAT`, `NGAY_THANG`)
- **NER_LABELS:** 19 BIO tags + O
- **Relation:** 7 types (`DUA_TREN`, `THAM_CHIEU`, `HET_HIEU_LUC`, `THAY_THE`, `SUA_DOI_BO_SUNG`, `HUONG_DAN`, `CHUA`)
- **Document types:** 7 classes (`hien_phap`, `bo_luat`, `luat`, `nghi_dinh`, `thong_tu`, `quyet_dinh`, `nghi_quyet`)

---

## 6. Dependencies from requirements.txt

| Dependency | Version | Purpose |
|------------|---------|---------|
| `python-dotenv` | >=1.0.0 | Environment config |
| `fastapi` | >=0.104.0 | Web backend |
| `uvicorn` | >=0.24.0 | ASGI server |
| `torch` | >=2.1.0 | Deep learning |
| `transformers` | >=4.36.0 | PhoBERT model |
| `scikit-learn` | >=1.3.0 | TF-IDF fallback |
| `numpy` | >=1.25.0 | Numerical ops |
| `py-vncorenlp` | >=0.1.4 | Vietnamese word segmentation |
| `networkx` | >=3.2.0 | Knowledge graph |
| `pyvis` | >=0.3.2 | Graph visualization |
| `rouge-score` | >=0.1.2 | Summarization evaluation |
| `pandas` | >=2.1.0 | Data processing |
| `matplotlib` | >=3.8.0 | Plotting |
| `tqdm` | >=4.66.0 | Progress bars |

**Note:** `peft` is used in `train_lora_phobert.py` but is not listed in `requirements.txt`.

---

## 7. New Scripts Deep Dive

### `scripts/parse_luat_ke_toan.py` (221 LOC)
- **Purpose:** Parses raw `luat_ke_toan_2025.txt` into structured JSON.
- **Hierarchy:** Chương -> Điều -> Khoản -> Điểm.
- **Dataclasses:** `Chuong`, `Dieu`, `Khoan`, `Diem`.
- **Output:** `data/processed/luat_ke_toan_2025_structured.json`.

### `scripts/generate_qa_dataset.py` (214 LOC)
- **Purpose:** Generates question-answer pairs from parsed law text (v1).
- **Templates:** 7 intent-specific template groups (default, dinh_nghia, doi_tuong, quyen_loi, nghia_vu, thu_tuc, thoi_han).
- **Features:** Auto-detects intent from article title keywords, extracts entities, builds BIO NER labels.
- **Output:** `data/processed/qa_ke_toan_train.json`.

### `scripts/generate_qa_dataset_v2.py` (347 LOC)
- **Purpose:** Expanded dataset generation with more templates and synonym replacement.
- **Enhancements over v1:**
  - 10 intent categories (added `hoi_hieu_luc`, `hoi_hinh_phat`, `hoi_sua_doi`).
  - Up to 14 templates per intent.
  - `synonym_replace()` for simple paraphrasing (`quy định` <-> `nêu rõ`, etc.).
  - `max_per_article` cap (default 12).
  - Intent distribution reporting with percentages.
- **Output:** `data/processed/qa_ke_toan_train_v2.json`.

### `scripts/train_lora_phobert.py` (272 LOC)
- **Purpose:** Fine-tune PhoBERT with PEFT LoRA adapters for multi-task learning (intent + NER).
- **LoRA Config:** `r=16`, `alpha=32`, `dropout=0.1`, target modules: `["query", "key", "value", "dense"]`.
- **Model:** `PhoBERTMultiTaskLoRA` — shared PhoBERT encoder + LoRA, with separate intent and NER heads.
- **Dataset:** `LawQADataset` loads `qa_ke_toan_train_v2.json`, aligns word-level NER labels to wordpiece tokens.
- **Training:** AdamW, linear warmup schedule, gradient clipping (1.0), cross-entropy loss for both tasks.
- **Checkpointing:** Saves best model by loss to `data/models/lora_ke_toan/best_model.pt`.
- **Metrics:** Reports intent accuracy and NER accuracy per epoch.

### `scripts/inference_lora.py` (234 LOC)
- **Purpose:** Load and run inference on the trained LoRA checkpoint.
- **Modes:**
  - **Demo:** Runs 5 sample questions automatically.
  - **Interactive (`-i`):** CLI REPL for user questions.
  - **Single (`-q`):** One-off question via CLI arg.
- **Functions:**
  - `load_trained_model()` — reconstructs intent classifier and NER tagger from checkpoint state dict.
  - `predict_intent()` — returns top-3 intent probabilities.
  - `predict_ner()` — returns extracted entities with cleaned text.
  - `search_answer()` — simple keyword-based retrieval from QA dataset.
- **Checkpoint:** Defaults to `data/models/lora_ke_toan/best_model.pt`.

---

## Additional Observations

- **Current model status:** The docs state "Heads defined, NOT trained (TF-IDF fallback active)." However, `data/models/lora_ke_toan/best_model.pt` exists, indicating the LoRA training script has been run successfully.
- **Test coverage:** Zero tests exist (`tests/` directory is empty). Docs mention planned pytest suite.
- **VnCoreNLP models:** Pre-downloaded in `vncorenlp/` (~500MB JAR + model files).
- **PEFT missing:** `train_lora_phobert.py` imports `peft` but it is absent from `requirements.txt`.
