# LegalAI — Project Overview & Product Development Requirements

## Project Overview

LegalAI is a Vietnamese legal AI assistant that combines 4 specialized modules, each mapped to a university AI course. The system answers legal questions by routing them through intent classification, document search, summarization, and knowledge graph reasoning.

**Core Principle**: All data flows through ALL 4 modules — no module operates in isolation.

| Module | Course | Key Algorithm | Role |
|--------|--------|---------------|------|
| Search | Algorithm Design | Inverted Index + BM25 + Trie + Levenshtein | Retrieve relevant legal documents |
| Summarizer | Python ML | PhoBERT sentence scorer + TextRank | Extract key sentences from documents |
| Chatbot | NLP | PhoBERT intent classifier (20 intents) + NER (9 entity types) | Understand question, route to modules, synthesize answer |
| Knowledge Graph | AI | NetworkX DiGraph + legal reasoning | Trace validity, amendments, relationships between documents |

## Product Development Requirements

### PDR-001: Legal Question Answering
- **Description**: User submits a Vietnamese legal question; system returns a structured answer with sources
- **Flow**: Question -> Chatbot (intent + NER) -> Search (BM25) -> Summarizer (extract key) -> KG (context) -> Chatbot (synthesis)
- **Acceptance Criteria**:
  - Intent classified with confidence score
  - Entities extracted from question
  - Top-K relevant documents retrieved
  - Summary of key sentences produced
  - Knowledge graph context appended when applicable
  - Final answer synthesized in Vietnamese

### PDR-002: Document Search
- **Description**: Full-text search across legal document corpus with BM25 ranking
- **Acceptance Criteria**:
  - Inverted index built from 600+ legal documents
  - BM25 ranking with configurable k1/b parameters
  - Boolean search (AND/OR/phrase)
  - Autocomplete via Trie prefix matching
  - Fuzzy search via Levenshtein edit distance
  - Score explainability (BM25 breakdown per term)

### PDR-003: Extractive Summarization
- **Description**: Summarize legal documents by selecting most important sentences
- **Acceptance Criteria**:
  - Three scoring components: relevance (40%), centrality (40%), position (20%)
  - TextRank PageRank on sentence similarity graph
  - PhoBERT-based sentence scoring (with TF-IDF fallback)
  - Configurable top-K sentence selection
  - Minimum gap constraint to avoid adjacent sentences

### PDR-004: Knowledge Graph Reasoning
- **Description**: Query legal document relationships and trace amendment chains
- **Acceptance Criteria**:
  - 9 entity types, 7 relation types extracted from corpus
  - Validity check with amendment/expiry tracing
  - Amendment chain traversal
  - Related document discovery (BFS, max depth 2)
  - Shortest reasoning path between entities
  - Interactive visualization via Pyvis (HTML export)

### PDR-005: Chatbot Orchestration
- **Description**: Central pipeline that routes questions to appropriate modules
- **Acceptance Criteria**:
  - 20 intent classes covering common legal queries
  - 9 entity types with BIO tagging
  - Intent-to-module routing table
  - Structured ChatResponse with answer, intent, entities, sources, reasoning

## Technical Constraints

| Constraint | Value |
|------------|-------|
| Python version | 3.11+ |
| Base model | vinai/phobert-base |
| Max sequence length | 256 tokens |
| Model status | Heads defined, NOT trained (TF-IDF fallback active) |
| Data size | 89,261 QA pairs + 600 legal documents |
| Knowledge graph | 20,382 nodes, 57,047 edges |
| Search index | 599 docs, 5,454 terms |

## Current Limitations

1. **No trained models**: PhoBERT heads use random weights; intent/NER/scorer fall back to TF-IDF heuristics
2. **No tests**: Zero test coverage
3. **No training scripts**: No fine-tuning pipeline for PhoBERT heads
4. **No notebooks**: No exploration/evaluation notebooks
5. **VnCoreNLP dependency**: Word segmentation requires py-vncorenlp download on first run

## Success Metrics

| Metric | Target |
|--------|--------|
| Intent classification accuracy | >85% on test set |
| NER F1 score | >75% on legal text |
| Search relevance (MRR@10) | >0.6 |
| Summary ROUGE-L | >0.4 |
| KG reasoning correctness | >90% on validity checks |
| API response time (p95) | <2s |