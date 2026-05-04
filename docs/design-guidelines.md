# LegalAI — Design Guidelines

## UI Design Principles

The Streamlit frontend (`src/ui.py`) follows a modular tab-based layout, with one tab per module:

| Tab | Purpose | Key Elements |
|-----|---------|--------------|
| Chatbot | Full pipeline interaction | Question input, structured response display, confidence badge |
| Search | Document retrieval | Query input, ranked results list, BM25 score display |
| Summarize | Document summarization | Document textarea, query input, summary output |
| Knowledge Graph | Graph reasoning | Doc ID input, query type selector, visualization iframe |

## Response Formatting

### Chat Response Display
```
[Intent Badge]    [Confidence Score]
[Answer Text]
---
Sources:
  1. [Doc Title] (Score: X.XX)
  2. [Doc Title] (Score: X.XX)
---
Entities:
  - [Type]: [Value]
  - [Type]: [Value]
---
Reasoning:
  1. [Step 1]
  2. [Step 2]
```

### Error States
- API unreachable: Red banner with retry button
- Search index not built: Yellow warning with build instructions
- Model not loaded: Gray badge indicating TF-IDF fallback mode

## Color Coding (Knowledge Graph Visualization)

| Entity/Relation Type | Color | Usage |
|----------------------|-------|-------|
| LUAT | Blue | Law nodes |
| THONG_TU | Green | Circular nodes |
| NGHI_DINH | Orange | Decree nodes |
| DIEU/KHOAN/DIEM | Gray | Article/clause/point nodes |
| CO_QUAN | Purple | Agency nodes |
| DUA_TREN | Solid line | Pursuant-to relation |
| THAY_THE | Dashed red | Supersedes relation |
| SUA_DOI_BO_SUNG | Dashed orange | Amends relation |
| HET_HIEU_LUC | Dotted gray | Expired relation |

## Accessibility

- Vietnamese text rendering: Ensure fonts support Unicode combining diacritics
- Pyvis visualization: Generated HTML is self-contained and works offline
- API error messages: Returned in English for debugging; UI messages in Vietnamese for users
