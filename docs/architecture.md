# LegalAI — Hệ thống Trợ lý Pháp luật Thông minh

## Kiến trúc: 1 hệ thống, 4 module xuyên suốt

```
User hỏi: "Thông tư 99 quy định gì về chế độ kế toán?"
    │
    ▼
┌──────────────────────────────────────────────────┐
│            Chatbot (NLP) — CỬA VÀO               │
│  PhoBERT fine-tuned: intent + NER                │
│  → hiểu câu hỏi, trích thực thể, điều phối       │
└──────────────────┬───────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────────┐
│ Search  │  │Knowledge │  │ Summarizer   │
│ (TT)    │  │ Graph(AI)│  │ (Python ML)  │
│ BM25+   │  │ reason   │  │ PhoBERT score│
│ Trie    │  │ +recommend│  │ extractive   │
└────┬────┘  └────┬─────┘  └──────┬───────┘
     │            │               │
     └────────────┼───────────────┘
                  ▼
         Chatbot tổng hợp câu trả lời
                  │
                  ▼
              User nhận
         "TT99 quy định chế độ
          kế toán theo 4 phần..."
```

## Data Source
- `yuiTC_sample.json`: 89,261 cặp hỏi đáp pháp luật VN
- `uts_vlc_processed.json`: 62.4MB văn bản pháp luật xử lý sẵn
- Crawl thêm từ thuvienphapluat.vn (scripts có sẵn từ legal-ai-agent)

## Phân công (3 người, 1 tháng)

| Người | Module | Nhiệm vụ chính |
|-------|--------|----------------|
| A | Chatbot + Knowledge Graph | PhoBERT fine-tune intent/NER, graph construction + reasoning |
| B | Search + Summarizer | Inverted index, BM25, Trie, PhoBERT extractive summarization |
| C | Data pipeline + Integration | ETL data, FastAPI, Streamlit, cross-module wiring |

## Model: PhoBERT-base (vinai/phobert-base)
- Fine-tune trên data pháp luật kế toán 2025
- 3 task heads: intent classifier, NER tagger, sentence scorer
- Phục vụ 3/4 module (chatbot, summarizer, knowledge graph)

## Tech Stack
- Python 3.11+
- PhoBERT (transformers + torch)
- FastAPI (backend)
- Streamlit (frontend)
- NetworkX (knowledge graph)
- Custom search (inverted index + BM25 + Trie)