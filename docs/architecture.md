# LegalAI — Hệ thống Trợ lý Pháp luật Thông minh

## Kiến trúc: 1 hệ thống, 4 module xuyên suốt

```
User hỏi: "Thông tư 99 quy định gì về chế độ kế toán?"
    │
    ▼
┌──────────────────────────────────────────────────┐
│            Chatbot (NLP) — CỬA VÀO             │
│  PhoBERT fine-tuned: intent + NER              │
│  → hiểu câu hỏi, trích thực thể, điều phối     │
└──────────────────┬─────────────────────────────┘
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
- `luat_ke_toan_2025.txt`: Luật Kế toán 2025 (raw text, optional)
- `luat_ke_toan_2025_structured.json`: Luật Kế toán 2025 đã phân cấp (Chương → Điều → Khoản → Điểm)
- `qa_ke_toan_train_v2.json`: Bộ QA tự sinh từ luật Kế toán (v2, có paraphrase)

## Phân công (3 người, 1 tháng)

| Người | Module | Nhiệm vụ chính |
|-------|--------|----------------|
| A | Chatbot + Knowledge Graph | PhoBERT fine-tune intent/NER (LoRA), graph construction + reasoning |
| B | Search + Summarizer | Inverted index, BM25, Trie, PhoBERT extractive summarization |
| C | Data pipeline + Integration | ETL data, FastAPI, Streamlit, cross-module wiring |

## Model: PhoBERT-base (vinai/phobert-base)

- Fine-tune trên data pháp luật kế toán 2025 bằng LoRA (r=16, alpha=32)
- 3 task heads: intent classifier, NER tagger, sentence scorer
- Phục vụ 3/4 module (chatbot, summarizer, knowledge graph)
- Checkpoint LoRA: `data/models/lora_ke_toan/best_model.pt`

## Tech Stack

- Python 3.11+
- PhoBERT (transformers + torch + peft)
- FastAPI (backend)
- Streamlit (frontend)
- NetworkX (knowledge graph)
- Custom search (inverted index + BM25 + Trie + Levenshtein)

## API Endpoints

| Endpoint | Phương thức | Mô tả |
|----------|-------------|-------|
| `/` | GET | Trạng thái hệ thống |
| `/health` | GET | Kiểm tra sức khỏe |
| `/search` | POST | Tìm kiếm BM25 |
| `/search/autocomplete` | POST | Gợi ý tự động Trie |
| `/search/explain` | POST | Giải thích điểm BM25 |
| `/summarize` | POST | Tóm tắt trích xuất |
| `/knowledge/query` | POST | Truy vấn KG |
| `/knowledge/stats` | POST | Thống kê KG |
| `/knowledge/visualize` | POST | Xuất HTML Pyvis |
| `/chat` | POST | Pipeline đầy đủ |
