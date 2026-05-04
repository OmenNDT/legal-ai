# LegalAI — System Architecture

## High-Level Data Flow

```
User Question (Vietnamese)
    |
    v
+------------------------------+
|  Chatbot (NLP Module)        |
|  1. Intent Classification    |  PhoBERT Head 1: 20 intents
|  2. Named Entity Recognition |  PhoBERT Head 2: 9 entity types (BIO)
|  3. Intent Routing           |  INTENT_ROUTING table -> which modules to call
+------------------------------+
    |                |                |
    | (if "search") | (if "kg")     | (if "summarize")
    v                v                v
+----------+  +--------------+  +---------------+
| Search   |  | Knowledge   |  | Summarizer    |
| (Alg.Des)|  | Graph (AI)  |  | (Python ML)   |
|          |  |              |  |               |
| BM25     |  | Validity     |  | PhoBERT score |
| Trie     |  | Amendments   |  | TextRank      |
| Levenshtein | Reasoning  |  | Position      |
+----------+  +--------------+  +---------------+
    |                |                |
    +--------+-------+--------+-------+
             |                |
             v                v
+------------------------------+
|  Chatbot Response Synthesis  |
|  - Combine search results   |
|  - Append summary           |
|  - Add KG reasoning steps   |
|  - Format Vietnamese answer |
+------------------------------+
    |
    v
Structured ChatResponse
(answer, intent, entities, sources, reasoning, summary)
```

## API Architecture

```
Streamlit UI (port 8501)
    |
    | HTTP requests
    v
FastAPI Backend (port 8000)
    |
    +-- GET  /              -> status check
    +-- GET  /health        -> health check
    +-- POST /search        -> BM25 ranked search
    +-- POST /search/autocomplete -> Trie prefix completion
    +-- POST /search/explain      -> BM25 score breakdown
    +-- POST /summarize    -> extractive summarization (TF-IDF fallback)
    +-- POST /knowledge/query    -> KG reasoning (validity/amendments/related/path)
    +-- POST /knowledge/stats    -> KG statistics
    +-- POST /knowledge/visualize -> Pyvis HTML export
    +-- POST /chat          -> full pipeline (intent -> NER -> search -> summarize -> KG -> answer)
```

**Request/Response Models** (defined in `src/app.py`):
- `QueryRequest`: `{question: str, top_k: int = 10}`
- `SearchRequest`: `{query: str, top_k: int = 10}`
- `SummarizeRequest`: `{document: str, query: str = "", top_k: int = 4}`
- `KGQueryRequest`: `{doc_id: str, query_type: str = "validity", target_id?: str}`

**Chat response fields**: `answer`, `intent`, `confidence`, `entities`, `sources`, `summary`, `reasoning`

**KG `query_type` values**: `validity`, `amendments`, `related`, `path`

## Module Internals

### Search Module
```
LegalSearchEngine
    |
    +-- InvertedIndex
    |   +-- Posting (doc_id, positions[])
    |   +-- Tokenization (Vietnamese diacritics preserved)
    |   +-- search(), search_and(), search_or(), search_not(), search_phrase()
    |   +-- save()/load() JSON serialization
    |
    +-- BM25 (over InvertedIndex)
    |   +-- IDF precomputation
    |   +-- score(D,Q) = sum(IDF(qi) * f(qi,D)*(k1+1) / (f(qi,D)+k1*(1-b+b*|D|/avgdl)))
    |   +-- explain() per-term score breakdown
    |
    +-- Trie
        +-- TrieNode (children, is_end, freq, doc_freq)
        +-- insert(), search_exact(), autocomplete()
        +-- Levenshtein edit distance DP
        +-- fuzzy_search() over vocabulary
```

### Summarizer Module
```
LegalSummarizer
    |
    +-- PhoBERTSentenceScorer (Head 3)
    |   Architecture: PhoBERT -> mean_pool(sent) -> [sent; doc; sent*doc] (2304)
    |   -> Linear(2304, 768) -> ReLU -> Dropout -> Linear(768,1)
    |   score_sentences(): per-sentence importance with document context
    |
    +-- TextRank
    |   build_similarity_matrix() -> cosine threshold pruning
    |   textrank() -> power iteration PageRank (damping=0.85)
    |   compute_position_scores() -> inverse position + first-sentence bonus
    |
    +-- Combined Scoring
        final = 0.4*relevance + 0.4*centrality + 0.2*position
        select_sentences() with min_gap constraint
```

### Knowledge Graph Module
```
LegalKnowledgeGraph (NetworkX DiGraph)
    |
    +-- Entity Extraction (regex)
    |   9 types: LUAT, THONG_TU, NGHI_DINH, DIEU, KHOAN, DIEM, CO_QUAN, KHAISUAT, NGAY_THANG
    |
    +-- Relation Extraction (regex)
    |   7 types: DUA_TREN, THAM_CHIEU, HET_HIEU_LUC, THAY_THE, SUA_DOI_BO_SUNG, HUONG_DAN, CHUA
    |
    +-- LegalReasoner
    |   check_validity() -> traces expiry, replacement, amendment edges
    |   trace_amendments() -> BFS over SUA_DOI_BO_SUNG/THAY_THE edges
    |   find_related() -> BFS depth-2 bidirectional
    |   find_reasoning_path() -> shortest_path via NetworkX
    |
    +-- Visualizer (Pyvis)
        visualize_graph() -> interactive HTML with type-based node/edge coloring
        visualize_amendment_chain() -> focused subgraph visualization
```

## Data Pipeline

### Core Pipeline
```
1. Download:     scripts/download_data.py -> data/raw/
2. Preprocess:   scripts/preprocess.py -> data/processed/
   - VnCoreNLP word segmentation
   - Legal reference extraction
   - NER label generation (BIO tags)
   - Train/val/test split (80/10/10)
3. Build Index:  scripts/build_index.py -> data/processed/search_index.json
4. Build KG:     scripts/build_graph.py -> data/processed/knowledge_graph.gpickle
5. Serve:        python -m uvicorn src.app:app --port 8000
6. UI:           streamlit run src/ui.py
```

### Optional LoRA Training Pipeline
```
1. Parse law:          scripts/parse_luat_ke_toan.py
   Input:  data/raw/luat_ke_toan_2025.txt
   Output: data/processed/luat_ke_toan_2025_structured.json
   Hierarchy: Chuong -> Dieu -> Khoan -> Diem

2. Generate QA v2:     scripts/generate_qa_dataset_v2.py
   Input:  luat_ke_toan_2025_structured.json
   Output: data/processed/qa_ke_toan_train_v2.json
   Features: 10 intent categories, up to 14 templates, synonym paraphrasing

3. Train LoRA:         scripts/train_lora_phobert.py
   Input:  qa_ke_toan_train_v2.json
   Output: data/models/lora_ke_toan/best_model.pt
   Config: r=16, alpha=32, dropout=0.1, target modules ["query","key","value","dense"]
   Tasks: intent classification + NER (multi-task)
   Optimizer: AdamW, linear warmup, gradient clipping (1.0), FP16

4. Inference:          scripts/inference_lora.py
   Modes: demo (5 samples), interactive (-i), single (-q)
   Functions: top-3 intent predictions, entity extraction, QA keyword retrieval
```

## Model Architecture (PhoBERT)

### Base PhoBERT + 3 Task Heads
```
vinai/phobert-base (frozen or fine-tuned)
    |
    +-- Head 1: Intent Classifier
    |   PhoBERT -> [CLS] (768) -> Dropout(0.1) -> Linear(768, 20)
    |   20 intent classes, cross-entropy loss
    |
    +-- Head 2: NER Tagger
    |   PhoBERT -> Dropout(0.1) -> Linear(768, 19)
    |   9 entity types * 2 (B/I) + 1 (O) = 19 labels
    |   BIO scheme, CrossEntropyLoss with ignore_index=-100
    |
    +-- Head 3: Sentence Scorer
        PhoBERT -> mean_pool -> concat[sent, doc, sent*doc] (2304)
        -> Linear(2304, 768) -> ReLU -> Dropout(0.3) -> Linear(768, 1)
        Binary relevance scoring (sentence in summary or not)
```

### LoRA Fine-Tuning Architecture
```
vinai/phobobert-base + PEFT LoRA adapters
    |
    +-- LoRA Config: r=16, alpha=32, dropout=0.1
    |   Target modules: ["query", "key", "value", "dense"]
    |
    +-- PhoBERTMultiTaskLoRA
        Shared encoder (PhoBERT + LoRA)
        +-- Intent Head: [CLS] -> Linear(768, 20)
        +-- NER Head: token-level -> Linear(768, 19)
        Loss = cross_entropy(intent) + cross_entropy(NER)
```

## Key Design Decisions

1. **Custom search**: No external search libs (Elasticsearch, Whoosh) — pure algorithm implementation for Algorithm Design course requirements
2. **Extractive summarization**: Not abstractive — selects existing sentences, avoiding hallucination risk in legal domain
3. **TF-IDF fallback**: Summarizer and chatbot degrade to sklearn TF-IDF when PhoBERT models are unavailable
4. **Lazy loading**: Models load on first request, not at import time, reducing startup cost
5. **NetworkX over Neo4j**: Simpler deployment, no external database dependency; sufficient for 20K nodes
6. **Regex entity extraction**: Faster and more predictable than ML-based extraction for structured legal text
7. **LoRA over full fine-tuning**: PEFT adapters reduce trainable parameters; single checkpoint serves both intent and NER tasks
