# LegalAI — Code Standards

## File Naming & Organization

- **Kebab-case** for filenames: `sentence_scorer.py`, `entity_extractor.py`, `build_index.py`
- **Snake_case** for Python modules and variables: `LegalSearchEngine`, `legal_documents`
- **One class per file** for major components; utility functions grouped by domain
- **Package structure**: each module has `__init__.py` and domain-specific files

## Module Boundaries

| Module | Package | Responsibility | External Deps |
|--------|---------|---------------|---------------|
| Search | `src/search/` | Inverted index, BM25, Trie, fuzzy search | None (pure Python) |
| Summarizer | `src/summarizer/` | Extractive summarization, TextRank | torch, transformers, sklearn |
| Chatbot | `src/chatbot/` | Intent classification, NER, orchestration | torch, transformers |
| Knowledge | `src/knowledge/` | Entity/relation extraction, graph, reasoning | networkx, pyvis |
| Common | `src/common/` | Config, data loading, text processing | py_vncorenlp |

**Rule**: No cross-module imports except through `common/` or `chatbot/pipeline.py` (orchestrator).

## Architecture Patterns

### Facade Pattern
Each module exposes a single entry-point class:
- `LegalSearchEngine` wraps `InvertedIndex`, `BM25`, `Trie`
- `LegalSummarizer` wraps `PhoBERTSentenceScorer`, `textrank` functions (from `src.summarizer.textrank`)
- `LegalChatbot` wraps `PhoBERTIntentClassifier`, `PhoBERTNERTagger`, routes to other modules
- `LegalKnowledgeGraph` wraps `LegalEntityExtractor`, `LegalRelationExtractor`

### Lazy Loading
Models load on first use via `_load_*()` methods:
```python
def _load_intent(self):
    if self._intent_classifier is None:
        self._intent_classifier = PhoBERTIntentClassifier()
        self._intent_tokenizer = AutoTokenizer.from_pretrained(...)
```

### Fallback Strategy
PhoBERT components degrade gracefully:
- Intent classifier: LoRA checkpoint loads if available; otherwise random weights (still runs, poor accuracy)
- Summarizer: `use_model=False` flag triggers TF-IDF cosine similarity fallback
- NER: untrained predictions produce noisy output unless LoRA checkpoint is loaded

## Coding Conventions

### Type Hints
All public methods use type hints:
```python
def search(self, query: str, top_k: int = 10) -> list[dict]:
def summarize(self, document: str, query: str = "", top_k: int = 4) -> dict:
```

### Dataclasses for Structured Results
```python
@dataclass
class ChatResponse:
    answer: str
    intent: str
    confidence: float
    entities: list[dict]
    sources: list[dict]
    reasoning: list[str]
    summary: Optional[str] = None
```

### Configuration via `common/config.py`
All constants in one place:
- Paths: `SEARCH_INDEX_PATH`, `KG_PATH`, model directories
- Hyperparameters: `BM25_K1`, `BM25_B`, `TEXTRANK_DAMPING`, `SUMMARY_WEIGHTS`
- Labels: `INTENT_LABELS`, `NER_LABELS`, `ENTITY_TYPES`, `RELATION_TYPES`

### Error Handling
- API endpoints return `{"error": "..."}` dicts, never raise unhandled exceptions
- Model loading wrapped in try/except with fallback
- Missing data files result in empty lists (graceful degradation)

### Vietnamese Text Processing
- **Always** segment text with VnCoreNLP before PhoBERT tokenization
- Use `src/common/text_processor.py: segment_text()` for word segmentation
- Vietnamese diacritics preserved in tokenization regex
- Legal reference extraction uses Vietnamese-specific patterns (Thong tu, Nghi dinh, Dieu, etc.)

## Testing Standards

**Current state**: Zero tests. `tests/` directory is empty.

**Planned**:
- **Framework**: pytest
- **Structure**: `tests/test_{module}_{component}.py`
- **Coverage target**: >80% for search, summarizer, knowledge modules
- **Integration tests**: FastAPI TestClient for all endpoints
- **Fixtures**: Small subset of QA data and legal docs in `tests/fixtures/`

## Dependency Management

Core dependencies in `requirements.txt`:
- Python 3.11+ required
- `transformers>=4.36` + `torch>=2.1` for PhoBERT
- `networkx>=3.2` for knowledge graph
- `scikit-learn>=1.3` for TF-IDF fallback
- `py-vncorenlp>=0.1.4` for word segmentation
- `fastapi>=0.104` + `uvicorn>=0.24` for API
- No external search library — all custom implementations

**Missing dependency**: `peft` is required by `train_lora_phobert.py` but not listed in `requirements.txt`. Install manually:
```bash
pip install peft
```
