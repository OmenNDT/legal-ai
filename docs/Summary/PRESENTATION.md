# Hệ thống Tóm tắt Văn bản Lai (Hybrid Summarization) trên bộ CUAD

> Tài liệu thuyết trình – mô tả chi tiết kiến trúc, luồng dữ liệu, thuật toán và cách triển khai. Phù hợp dùng cho slide báo cáo và phần demo.

---

## 1. Bài toán

- **Dữ liệu**: CUAD v1 – 510 hợp đồng pháp lý tiếng Anh (`data/full_contract_txt/*.txt`).
  - Trung bình ~7.861 từ/file, tối đa ~47.733 từ (≈ 50–60 trang).
  - Kèm theo `master_clauses.csv` (83 cột) chứa các điều khoản quan trọng do luật sư gán nhãn → dùng làm **bản tóm tắt tham chiếu** (reference) để chấm ROUGE.
- **Mục tiêu**: sinh ra một bản tóm tắt ngắn, mượt mà cho từng hợp đồng dù file dài tới mức nào.
- **Thách thức**: BART/T5 chỉ nhận ≤ 1024 token. Đẩy thẳng file 47k từ vào sẽ tràn bộ nhớ và mất ngữ cảnh.

---

## 2. Hướng tiếp cận: Hybrid (Extractive → Abstractive)

```
[Hợp đồng dài 7k–47k từ]
        │
        ▼  ① Preprocess (clean + tách câu)
   List câu sạch
        │
        ▼  ② Extractive (TF-IDF / TextRank / KMeans / Ensemble)
   Top 20% câu cốt lõi (đảm bảo ≤ ~1024 token)
        │
        ▼  ③ Abstractive (BART-large-CNN)
   Bản viết lại 150–300 từ
        │
        ▼  ④ Evaluate (ROUGE-1/2/L, BERTScore)
   Báo cáo chất lượng
```

**Tại sao Hybrid?**
- Mô hình Deep Learning xử lý văn bản dài rất tốn RAM/VRAM. Nếu cắt cụt đầu vào, sẽ mất ý ở cuối.
- Trước tiên dùng thuật toán nhẹ (extractive) để **lọc 20% câu chứa thông tin cốt lõi**, rồi mới giao cho BART viết lại.
- Lợi ích: tiết kiệm tài nguyên + giữ được phong cách trừu tượng tự nhiên.

---

## 3. Kiến trúc thư mục

```
text_sumarisation/
├── data/                                # CUAD gốc
│   ├── full_contract_txt/               # 510 .txt
│   └── master_clauses.csv               # 83 cột clause → reference
│
├── backend/                             # Python OOP
│   ├── config/settings.py               # cấu hình chung
│   ├── preprocess/                      # ① loader + cleaner + splitter
│   ├── extractive/                      # ② tfidf/textrank/kmeans/ensemble
│   ├── abstractive/                     # ③ chunker + BART
│   ├── hybrid/pipeline.py               # orchestrator
│   ├── evaluate/                        # ④ reference + ROUGE + BERTScore
│   ├── training/                        # fine-tune BART trên worker1 (GPU)
│   ├── app/                             # Flask REST API (port 9020)
│   └── utils/                           # logger, timer, IO
│
├── frontend/                            # React 19 + AntD 6 + Tailwind 4
│   └── src/components/TabSummarization.jsx
│
├── deploy/                              # fabfile + requirements + run script
│   ├── fabfile.py                       # sync code → worker1 và start training
│   └── requirements*.txt
│
└── outputs/                             # ROUGE, model fine-tune, summary cache
```

---

## 4. Luồng hoạt động chi tiết của thuật toán

### Bước ① — Preprocess
File `backend/preprocess/`. Mục tiêu: từ chuỗi `raw_text` → danh sách câu sạch.

1. **ContractLoader** – đọc file `.txt` thành object `Contract{doc_id, raw_text, word_count}`.
2. **TextCleaner** – chuẩn hoá Unicode NFKC, loại `Page X of Y`, dòng `Source:`, gộp khoảng trắng/xuống dòng dư thừa.
3. **SentenceSplitter** – ưu tiên `nltk.sent_tokenize` (punkt), fallback regex. Lọc câu < 5 từ hoặc > 80 từ để bỏ tiêu đề/dòng nhiễu.

Output: `List[Sentence(idx, text, word_count)]`.

### Bước ② — Extractive (3 thuật toán, có thể ensemble)

Tất cả extractor kế thừa `BaseExtractor`. Hàm `extract()` chấm điểm câu, lấy top-K (`top_k_ratio=0.2` mặc định) rồi **giữ lại đúng thứ tự gốc** để bản tóm tắt mạch lạc.

**(a) TF-IDF** – `tfidf_extractor.py`
- Mỗi câu là một "document".
- `TfidfVectorizer(stop_words='english', ngram=(1,2))` → ma trận TF-IDF.
- Điểm câu = (tổng TF-IDF của câu) / (số từ trong câu).
- Trực giác: câu chứa nhiều từ khoá quan trọng & hiếm thì điểm cao.

**(b) TextRank** – `textrank_extractor.py`
- Vector hoá câu bằng TF-IDF rồi tính **cosine similarity** giữa mọi cặp câu.
- Dựng đồ thị có trọng số (cạnh = độ tương đồng).
- Chạy `networkx.pagerank(damping=0.85)` → câu trung tâm (có nhiều "vote" từ các câu khác) được điểm cao.
- Trực giác: tương tự PageRank của Google, câu nào "được nhiều câu khác đồng tình" thì quan trọng.

**(c) K-Means** – `kmeans_extractor.py`
- Embed câu bằng `sentence-transformers/all-MiniLM-L6-v2` (384 chiều).
- Gom thành K cụm (K = top-K cần giữ).
- Câu càng **gần tâm cụm** càng đại diện → điểm = `1 / (1 + distance_to_centroid)`.
- Trực giác: mỗi cụm = một chủ đề con; lấy 1 đại diện của mỗi cụm để tóm tắt đa chủ đề.

**(d) Ensemble** – `ensemble.py`
- Min-max normalize điểm của từng extractor rồi cộng theo trọng số `[TF-IDF=1.0, TextRank=1.5, KMeans=1.0]` (TextRank được ưu tiên vì thường mạnh nhất).

Output mỗi extractor: `ExtractResult{method, sentences[], scores[]}` – list câu top-K theo thứ tự gốc.

### Bước ③ — Abstractive (BART)

File `backend/abstractive/`.

1. **LongDocChunker** – nếu list câu sau extractive vẫn dài hơn 1024 token (do file gốc 47k từ), chia thành các chunk ≤ 1024 token với overlap 50 token để giữ ngữ cảnh.
2. **BartSummarizer** – wrap `facebook/bart-large-cnn`. Mỗi chunk:
   - Tokenize, truncate 1024.
   - `model.generate(num_beams=4, length_penalty=2.0, no_repeat_ngram_size=3, max_length=256)`.
3. Nếu có nhiều chunk → ghép các bản tóm tắt → nếu tổng vẫn > 1024 thì tóm tắt thêm một vòng nữa (hierarchical).

Output: `AbstractiveResult{text, chunks[], chunk_summaries[]}`.

### Bước ④ — Hybrid Pipeline (orchestrator)

`backend/hybrid/pipeline.py`. `HybridPipeline.run(raw_text)` lần lượt: clean → split → extract → (abstract) và đo thời gian từng bước qua `Timer`. Trả về `HybridResult.to_dict()` để gửi qua API.

### Bước ⑤ — Evaluate

`backend/evaluate/`.

- **ReferenceBuilder**: đọc `master_clauses.csv`, gộp các clause của từng doc lại (loại trùng) → "tóm tắt vàng" để chấm điểm.
- **RougeEvaluator**: ROUGE-1, ROUGE-2, ROUGE-L (precision/recall/F1).
- **BertScoreEvaluator**: BERTScore với `roberta-large` cho đánh giá nghĩa.
- **EvalRunner**: chạy benchmark trên toàn bộ 510 file, lưu kết quả vào `outputs/eval/rouge_*.json`.

---

## 5. Fine-tune BART trên CUAD (worker1, RTX 3090 24GB)

File `backend/training/`.

- **CuadDatasetBuilder**: với mỗi doc, `input = câu trích xuất bằng TextRank`, `target = clauses gộp từ CSV`. Chia train/val/test 80/10/10 (seed=42).
- **BartFineTuner** (Seq2SeqTrainer): epochs=3, batch=2, grad_accum=8 (eff batch=16), lr=3e-5, fp16, beams=4. Có cờ `--use_lora` để giảm VRAM bằng LoRA (r=16) khi cần.
- **run_train.py**: entrypoint CLI. Cache dataset vào `outputs/eval/dataset_cuad.json` để không build lại.

Triển khai từ máy local sang worker1 qua Fabric:

```bash
# trên máy local
cd text_sumarisation/deploy
fab sync               # rsync code lên worker1
fab setup              # tạo venv, cài torch CUDA 12.1 + deps
fab gpu                # check nvidia-smi và torch.cuda.is_available()
fab train --epochs=3 --batch=2 --grad-accum=8
fab tail               # xem log train
fab pull-model         # khi xong, kéo model về local
```

---

## 6. Backend Flask (port 9020)

`backend/app/server.py` → `python -m backend.app.server`.

| Endpoint | Method | Mô tả |
|---|---|---|
| `/api/health` | GET | Trạng thái + device + model |
| `/api/documents` | GET | Danh sách doc_id (có `q`, `page`, `page_size`) |
| `/api/documents/<doc_id>` | GET | Nội dung 1 doc + reference |
| `/api/extract` | POST | Chỉ chạy extractive (nhanh, không cần GPU) |
| `/api/summarize` | POST | Chạy đủ pipeline lai, kèm ROUGE nếu là doc CUAD |
| `/api/eval/run` | POST | Benchmark trên toàn bộ/limit doc, ghi kết quả ra file |

Request mẫu `/api/summarize`:
```json
{
  "doc_id": "2ThemartComInc_19990826_..._Agency Agreement",
  "extractor": "textrank",
  "use_abstractive": true
}
```

Response chứa: `extractive.{sentences[], scores[], text}`, `abstractive.{text, chunk_summaries[]}`, `timings`, `rouge` (nếu có reference).

---

## 7. Frontend tab "Tóm tắt văn bản"

`frontend/src/components/TabSummarization.jsx`. Bố cục 3 cột điều khiển + panel kết quả.

**Cột 1 – Nguồn đầu vào**
- Chọn 1 trong 510 doc CUAD (autocomplete tìm theo tên file).
- Hoặc dán/upload `.txt` tự do.

**Cột 2 – Thuật toán**
- Dropdown chọn `TF-IDF | TextRank | KMeans | Ensemble`.
- Switch bật/tắt bước Abstractive (BART).

**Cột 3 – Luồng xử lý**
- Hiển thị tiến trình 4 bước (clean → split → extract → abstract), highlight bước đang chạy.

**Panel kết quả**
- 4 thẻ thống kê: số từ gốc, số câu sau split, số câu giữ lại, tỉ lệ nén.
- Danh sách câu được trích kèm score.
- Bản viết lại của BART (có thể xem chi tiết từng chunk).
- ROUGE-1/2/L + precision/recall nếu chạy trên doc CUAD.
- Nút xuất kết quả `.txt`.

Frontend gọi backend qua axios `/api/*` → Vite proxy → Flask `localhost:9020`.

---

## 8. Cách chạy local (demo)

```bash
# Backend
cd text_sumarisation
pip install -r deploy/requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
PYTHONPATH=$PWD API_PORT=9020 python -m backend.app.server

# Frontend
cd frontend
npm install
npm run dev    # mở http://localhost:5173/legal-ai/
```

Vào tab "Tóm tắt văn bản", chọn doc, bấm "Chạy pipeline lai".

---

## 9. Slide checklist (để dán nhanh vào slide)

1. **Slide 1 – Bài toán**: 510 hợp đồng dài, cần tóm tắt, không có gold summary, BART chỉ chịu 1024 token.
2. **Slide 2 – Giải pháp**: Hybrid pipeline 4 bước (sơ đồ ở §2).
3. **Slide 3 – Preprocess**: clean Unicode, regex bỏ noise, tách câu NLTK, lọc 5–80 từ.
4. **Slide 4 – Extractive**: 3 thuật toán + ensemble; nêu trực giác từng thuật toán; lý do chọn top 20%.
5. **Slide 5 – Abstractive**: BART-large-CNN; chunk 1024 token, beam=4, hierarchical khi quá dài.
6. **Slide 6 – Reference từ CUAD**: tận dụng `master_clauses.csv` làm "tóm tắt vàng" – điểm độc đáo.
7. **Slide 7 – Evaluate**: ROUGE-1/2/L + BERTScore.
8. **Slide 8 – Fine-tune**: input = câu TextRank, target = clauses; LoRA tuỳ chọn; chạy trên RTX 3090.
9. **Slide 9 – Backend Flask + Frontend React**: kiến trúc service, các API.
10. **Slide 10 – Demo**: chạy 1 doc thật, chỉ thanh tiến trình, kết quả, ROUGE.
11. **Slide 11 – Hạn chế & hướng tiếp**: BART pretrained trên tin tức nên hơi gượng với legal → thử LED/Pegasus-BillSum; có thể thêm chấm điểm hậu kiểm bằng QA.

---

## 10. Phụ lục: chú thích các tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `TOP_K_RATIO` | 0.2 | Tỉ lệ câu extractive giữ lại |
| `MIN_SENT_LEN` | 5 | Bỏ câu < N từ (loại tiêu đề ngắn) |
| `MAX_SENT_LEN` | 80 | Bỏ câu > N từ (loại đoạn dài bị tách nhầm) |
| `BART_MAX_INPUT` | 1024 | Số token tối đa BART nhận |
| `BART_MAX_OUTPUT` | 256 | Số token sinh ra tối đa |
| `BART_NUM_BEAMS` | 4 | Beam search width |
| `CHUNK_OVERLAP` | 50 | Số câu chồng lấn giữa các chunk |

