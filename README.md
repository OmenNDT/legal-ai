# LegalAI — Vietnamese Legal AI Assistant

A modular legal question-answering platform for Vietnamese law, combining custom search, PhoBERT-based NLP, extractive summarization, and knowledge graph reasoning.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| NLP Model | vinai/phobert-base + LoRA adapters |
| Search | Custom InvertedIndex + BM25 + Trie + Levenshtein |
| Knowledge Graph | NetworkX DiGraph + Pyvis |
| Word Segmentation | VnCoreNLP |

## Module Overview

| Module | Course | Key Algorithm | Responsibility |
|--------|--------|---------------|----------------|
| Search | Algorithm Design | InvertedIndex + BM25 + Trie + Levenshtein | Retrieve relevant legal documents |
| Chatbot | NLP | PhoBERT intent classifier (20 intents) + NER (9 entity types) | Understand question, route to modules, synthesize answer |
| Summarizer | Python ML | PhoBERT sentence scorer + TextRank | Extract key sentences from documents |
| Knowledge Graph | AI | NetworkX DiGraph + legal reasoning | Trace validity, amendments, relationships between documents |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install peft  # required for LoRA training, missing from requirements.txt

# 2. Download data
python scripts/download_data.py

# 3. Preprocess
python scripts/preprocess.py

# 4. Build search index and knowledge graph
python scripts/build_index.py

# 5. Start API
python -m uvicorn src.app:app --port 8000

# 6. Start UI
streamlit run src/ui.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Platform status |
| `/health` | GET | Health check |
| `/search` | POST | BM25 ranked search |
| `/search/autocomplete` | POST | Trie prefix suggestions |
| `/search/explain` | POST | BM25 score breakdown |
| `/summarize` | POST | Extractive summarization |
| `/knowledge/query` | POST | KG reasoning (validity, amendments, related, path) |
| `/knowledge/stats` | POST | KG statistics |
| `/knowledge/visualize` | POST | Pyvis HTML export |
| `/chat` | POST | Full chatbot pipeline |

## Data Pipeline

```
download_data.py  ->  preprocess.py  ->  build_index.py  ->  serve
     |                    |                   |
  raw/               processed/          search_index.json
  yuiTC_sample.json  qa_train.json       knowledge_graph.gpickle
  uts_vlc_processed.json legal_docs.json
```

Optional pipeline for LoRA fine-tuning on Luat Ke toan 2025:

```
parse_luat_ke_toan.py -> generate_qa_dataset_v2.py -> train_lora_phobert.py -> inference_lora.py
```

## Development Setup

- Python 3.11+
- VnCoreNLP auto-downloads on first run (~500MB)
- PhoBERT-base requires ~1.5GB GPU memory for training; CPU inference works

## Project Structure

```
legal-ai/
├── src/               # Core source (search, chatbot, summarizer, knowledge, common)
├── scripts/           # Data pipeline, training, inference scripts
├── data/              # raw/, processed/, models/
├── docs/              # Documentation
├── tests/             # Empty — tests not yet implemented
└── requirements.txt
```

## Key Stats

- 27 Python files, ~4,595 LOC
- 89,261 QA pairs + 600 legal documents
- 20 intent classes, 9 entity types (BIO), 7 relation types
- Knowledge graph: ~20,382 nodes, ~57,047 edges
- Search index: ~599 docs, ~5,454 terms

## Documentation

See `./docs/` for full documentation:
- [Project Overview & PDR](./docs/project-overview-pdr.md)
- [System Architecture](./docs/system-architecture.md)
- [Code Standards](./docs/code-standards.md)
- [Deployment Guide](./docs/deployment-guide.md)
- [Project Roadmap](./docs/project-roadmap.md)
