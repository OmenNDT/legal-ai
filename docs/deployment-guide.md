# LegalAI — Deployment Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required for modern type hints |
| pip | Latest | Package manager |
| Git | Latest | For data download script |
| VnCoreNLP | Java-based | Auto-downloads on first use (~500MB) |

## Installation

### 1. Clone and Install Dependencies

```bash
git clone <repo-url> legal-ai
cd legal-ai
pip install -r requirements.txt
```

This installs: FastAPI, uvicorn, torch, transformers, scikit-learn, networkx, pyvis, py-vncorenlp, and other dependencies.

### 2. Download Data

```bash
python scripts/download_data.py
```

Downloads from GitHub:
- `data/raw/yuiTC_sample.json` — 89,261 Vietnamese legal QA pairs
- `data/raw/uts_vlc_processed.json` — 600 processed legal documents

### 3. Preprocess Data

```bash
python scripts/preprocess.py
```

Processes:
- Word segmentation via VnCoreNLP (first run downloads model ~500MB)
- Legal reference extraction
- NER BIO label generation
- Train/val/test split (80/10/10)

Outputs:
- `data/processed/qa_train.json`, `qa_val.json`, `qa_test.json`
- `data/processed/legal_docs.json`

### 4. Build Search Index and Knowledge Graph

```bash
python scripts/build_index.py
```

Outputs:
- `data/processed/search_index.json` — serialized InvertedIndex
- `data/processed/knowledge_graph.gpickle` — NetworkX DiGraph

Expected stats:
- Search: ~599 docs, ~5,454 terms
- KG: ~20,382 nodes, ~57,047 edges

### 5. Start API Server

```bash
python -m uvicorn src.app:app --port 8000 --reload
```

The server loads search index and knowledge graph on startup. Verify:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/
# {"name":"LegalAI Platform","modules":["chatbot","search","summarizer","knowledge"],...}
```

### 6. Start Streamlit UI

```bash
streamlit run src/ui.py
```

Opens at http://localhost:8501 with 4 tabs: Chatbot, Search, Summarize, Knowledge Graph.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | BM25 ranked search: `{query, top_k}` |
| `/search/autocomplete` | POST | Trie prefix suggestions: `?prefix=...&max_results=...` |
| `/search/explain` | POST | BM25 score breakdown: `?query=...&doc_id=...` |
| `/summarize` | POST | Extractive summarization: `{document, query, top_k}` |
| `/knowledge/query` | POST | KG reasoning: `{doc_id, query_type, target_id?}` |
| `/knowledge/stats` | GET | KG statistics |
| `/knowledge/visualize` | POST | Pyvis HTML export |
| `/chat` | POST | Full pipeline: `{question, top_k}` |
| `/health` | GET | Health check |

**Chat response fields**: `answer`, `intent`, `confidence`, `entities`, `sources`, `summary`, `reasoning`

**KG query_type values**: `validity`, `amendments`, `related`, `path`

## Troubleshooting

### VnCoreNLP Download Fails
VnCoreNLP requires Java runtime. Install JRE/JDK first:
```bash
# Ubuntu/Debian
sudo apt install default-jre

# macOS
brew install openjdk
```

### Out of Memory on Model Loading
PhoBERT-base requires ~1.5GB GPU memory. For CPU-only:
- Reduce batch size in config (`TRAIN_BATCH_SIZE=8`)
- Use `FP16=True` for mixed precision training
- The summarizer has `use_model=False` fallback (TF-IDF only)

### Search Index / KG Not Found
Run `python scripts/build_index.py` first. The API returns `{"error": "Search engine not loaded. Run build_index.py first."}` if missing.

### Port Already in Use
```bash
lsof -i :8000          # macOS/Linux
netstat -ano | findstr :8000  # Windows
python -m uvicorn src.app:app --port 8001  # alternate port
```

## Project Configuration

All configurable values are in `src/common/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `BM25_K1` | 1.5 | BM25 term frequency saturation |
| `BM25_B` | 0.75 | BM25 document length normalization |
| `FUZZY_MAX_DIST` | 2 | Levenshtein max edit distance |
| `TEXTRANK_DAMPING` | 0.85 | PageRank damping factor |
| `SUMMARY_WEIGHTS` | {relevance: 0.4, centrality: 0.4, position: 0.2} | Sentence scoring weights |
| `MAX_SEQ_LENGTH` | 256 | PhoBERT max token length |
| `TRAIN_BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-5 | PhoBERT learning rate |
| `NUM_EPOCHS` | 30 | Training epochs |
| `EARLY_STOPPING_PATIENCE` | 5 | Early stopping patience |

## Data Directory Structure

```
data/
├── raw/                          # Source data (from download_data.py)
│   ├── yuiTC_sample.json        # 89,261 QA pairs
│   └── uts_vlc_processed.json   # 600 legal documents
├── processed/                    # Built artifacts (from preprocess.py + build_index.py)
│   ├── search_index.json         # Serialized InvertedIndex
│   ├── knowledge_graph.gpickle   # NetworkX DiGraph pickle
│   ├── legal_docs.json           # Preprocessed documents
│   ├── qa_train.json             # Train split
│   ├── qa_val.json               # Validation split
│   └── qa_test.json              # Test split
├── models/                       # Trained model weights (empty until training)
│   ├── intent_classifier/
│   ├── ner_tagger/
│   └── sentence_scorer/
└── embeddings/                   # VnCoreNLP model files (auto-downloaded)
```