# Hệ thống Tóm tắt Văn bản Lai (Hybrid Summarization) trên bộ CUAD

> Tài liệu thuyết trình – mô tả **chi tiết từng dòng** luồng hoạt động của hệ thống: từ cấu hình toàn cục, qua bốn bước preprocess/extractive/abstractive/evaluate, qua Flask REST API, đến frontend React. Có thể dán thẳng từng mục vào slide.

---

## 1. Bài toán & dữ liệu

- **Dữ liệu**: CUAD v1 – 510 hợp đồng pháp lý tiếng Anh nằm tại `data/full_contract_txt/*.txt`.
  - Trung bình ~7.861 từ/file, tối đa ~47.733 từ (≈ 50–60 trang).
  - Kèm `data/master_clauses.csv` (83 cột clause do luật sư gán nhãn) → dùng làm **bản tóm tắt tham chiếu** (reference) khi chấm ROUGE.
- **Mục tiêu**: với mỗi hợp đồng, sinh ra một bản tóm tắt 150–300 từ, súc tích, vẫn giữ điều khoản quan trọng (luật áp dụng, thời hạn, các bên, nghĩa vụ chính).
- **Thách thức cốt lõi**: BART / T5 chỉ nhận **≤ 1024 token** đầu vào. Đẩy thẳng một file 47k từ → tràn VRAM, mất hoàn toàn ngữ cảnh nửa sau.

→ Hướng giải: **Hybrid = Extractive (lọc câu) → Abstractive (viết lại bằng BART)**.

---

## 2. Sơ đồ luồng dữ liệu tổng thể

```
[Hợp đồng dài 7k–47k từ raw_text]
        │
        ▼  ① TextCleaner.clean()  (preprocess/cleaner.py)
   Văn bản đã sạch (NFKC, bỏ Page/Source, gom whitespace)
        │
        ▼  ② SentenceSplitter.split()  (preprocess/splitter.py)
   List[Sentence] – câu 5–80 từ, đánh idx
        │
        ▼  ③ Extractor.extract()  (extractive/*)
   List câu top-20% (giữ thứ tự gốc) + score
        │
        ▼  ④ LongDocChunker.chunk_by_sentences()  (abstractive/chunker.py)
   N chunk, mỗi chunk ≤ 1024 token, overlap 50 câu
        │
        ▼  ⑤ BartSummarizer._summarize_chunk()  (abstractive/bart_summarizer.py)
   N bản tóm tắt nhỏ → merge
        │
        ▼  Nếu merged > 1024 token  → BART tóm tắt LẠI lần nữa (hierarchical)
        │
        ▼  ⑥ Final summary (150–300 từ)
        │
        ▼  ⑦ RougeEvaluator + BertScoreEvaluator (nếu có reference)
   Báo cáo ROUGE-1/2/L (+ BERTScore khi chạy batch eval)
```

> Lưu ý: `LongDocChunker.chunk_by_sentences()` **luôn** được gọi sau bước extractive (không có nhánh if/else). Khi văn bản ngắn, nó trả về 1 chunk duy nhất → BART chỉ chạy 1 lần. Khi dài, mới phát sinh nhiều chunk + vòng tóm tắt thứ hai.

---

## 3. Kiến trúc thư mục

```
text_sumarisation/
├── data/
│   ├── full_contract_txt/             # 510 file .txt (đầu vào)
│   └── master_clauses.csv             # 83 cột clause → reference
│
├── backend/                           # Python OOP, mỗi module 1 trách nhiệm
│   ├── config/settings.py             # Settings + get_settings() (đọc env)
│   ├── preprocess/
│   │   ├── loader.py                  # ContractLoader, dataclass Contract
│   │   ├── cleaner.py                 # TextCleaner (regex + NFKC)
│   │   └── splitter.py                # SentenceSplitter (nltk punkt + fallback regex)
│   ├── extractive/
│   │   ├── base.py                    # BaseExtractor + ExtractResult
│   │   ├── tfidf_extractor.py         # TF-IDF + chuẩn hoá độ dài
│   │   ├── textrank_extractor.py      # cosine sim + nx.pagerank
│   │   ├── kmeans_extractor.py        # SBERT MiniLM + KMeans
│   │   └── ensemble.py                # min-max + weighted sum
│   ├── abstractive/
│   │   ├── chunker.py                 # LongDocChunker
│   │   └── bart_summarizer.py         # BartSummarizer + AbstractiveResult
│   ├── hybrid/pipeline.py             # HybridPipeline orchestrator + Factory
│   ├── evaluate/
│   │   ├── reference_builder.py       # gộp clause từ CSV
│   │   ├── rouge_scorer.py            # ROUGE-1/2/L (P/R/F)
│   │   ├── bert_scorer.py             # BERTScore roberta-large
│   │   └── runner.py                  # EvalRunner: lặp 510 file, ghi JSON
│   ├── training/
│   │   ├── dataset_builder.py         # CuadDatasetBuilder (input=TextRank, target=clauses)
│   │   ├── trainer.py                 # BartFineTuner + LoRA tuỳ chọn + EarlyStopping
│   │   └── run_train.py               # entrypoint CLI fine-tune
│   ├── utils/                         # Timer, Logger, JsonIO, PickleIO
│   └── app/                           # Flask
│       ├── server.py                  # create_app, serve frontend dist, health
│       ├── state.py                   # AppState singleton: loader, BART, cache pipeline
│       ├── schemas.py                 # SummarizeRequest + validate
│       ├── auth.py                    # /api/auth/* (PostgreSQL + bcrypt + JWT)
│       └── routes/                    # summarize, extract, documents, eval
│
├── frontend/                          # React 19 + AntD 6 + Tailwind 4 + Vite 8
│   └── src/components/TabSummarization.jsx
│
├── deploy/
│   ├── fabfile.py                     # sync/setup/gpu/train/tail/pull-model/serve/stop
│   ├── requirements.txt               # CPU local
│   ├── requirements-gpu.txt           # worker1 (CUDA 12.1)
│   └── run_backend.sh
│
└── outputs/
    ├── bart-cuad/, bart-cuad-v3/      # checkpoint fine-tune đã pull về
    └── eval/                          # rouge_*.json + dataset_cuad.json cache
```

---

## 4. Cấu hình toàn cục — `backend/config/settings.py`

Tất cả tham số được tập trung vào class `Settings`. Một số mặc định quan trọng:

| Field | Giá trị | Mô tả |
|---|---|---|
| `TXT_DIR` | `data/full_contract_txt/` | Folder chứa 510 hợp đồng |
| `CSV_FILE` | `data/master_clauses.csv` | Reference cho ROUGE |
| `TOP_K_RATIO` | `0.2` | Giữ 20% câu sau extractive |
| `MIN_SENT_LEN` / `MAX_SENT_LEN` | `5` / `80` | Lọc câu nhiễu (tiêu đề ngắn, đoạn dính nhau) |
| `BART_MODEL` | `facebook/bart-large-cnn` | Có thể override bằng env `BART_MODEL` để trỏ về checkpoint fine-tune |
| `BART_MAX_INPUT` / `BART_MAX_OUTPUT` / `BART_MIN_OUTPUT` | `1024` / `256` / `80` | Giới hạn token vào/ra |
| `BART_NUM_BEAMS` | `4` | Beam search width |
| `SBERT_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding cho KMeans |
| `CHUNK_OVERLAP` | `50` | Số câu chồng giữa các chunk |
| `API_PORT` | `9020` | Override bằng env `API_PORT` |
| `device()` | tự dò | `cuda` nếu `torch.cuda.is_available()` else `cpu` |
| `ensure_dirs()` | – | Tạo trước `outputs/{extractive,abstractive,hybrid,eval}`, `logs/`, `backend/cache/{models,summaries}` |

`get_settings()` còn đọc 2 biến môi trường (`API_PORT`, `BART_MODEL`) rồi gọi `ensure_dirs()` trước khi trả về.

---

## 5. Bước ① — Preprocess (làm sạch + tách câu)

### 5.1. Load file — `preprocess/loader.py`

- `Contract` (dataclass): `doc_id`, `file_path`, `raw_text`, `word_count`, `meta`.
- `ContractLoader.list_ids()` → liệt kê tất cả `*.txt` (bỏ đuôi), `sorted`.
- `ContractLoader.load_one(doc_id)` → đọc UTF-8, tính `word_count = len(text.split())`. Đây là số được hiển thị trên thẻ "Số từ gốc" của frontend.
- `iter_all()` là **generator** để tiết kiệm RAM (510 file × vài chục KB là OK, nhưng training/eval thường stream).

### 5.2. Làm sạch — `preprocess/cleaner.py`

`TextCleaner.clean(text)` thực hiện tuần tự:

1. `unicodedata.normalize("NFKC", text)` — gom các code-point Unicode tương đương (ví dụ chữ "fi" ligature → "fi").
2. Lọc ký tự không in được, chỉ giữ `\n\t` và `isprintable()`.
3. Regex bỏ noise đặc trưng PDF→text của CUAD:
   - `PAGE_NUM_RE` = `\bPage\s+\d+\s+of\s+\d+\b`
   - `SOURCE_RE` = `Source:\s*[^\n]+`
4. Gộp dấu chấm liên tục `\.{3,}` → `.`, gộp khoảng trắng `[ \t]+` → ` `, gộp xuống dòng `\n{3,}` → `\n\n`.
5. `keep_unicode=True` (mặc định) → giữ ký tự không phải ASCII (rất hiếm trong CUAD nhưng có).

Kết quả: chuỗi text "phẳng", dễ tách câu.

### 5.3. Tách câu — `preprocess/splitter.py`

- `Sentence` (dataclass): `idx`, `text`, `word_count`.
- `SentenceSplitter` ưu tiên `nltk.sent_tokenize` (model punkt), fallback regex `(?<=[\.!?])\s+(?=[A-Z0-9\"'(\[])` nếu NLTK chưa cài.
- `_try_load_nltk()` tự `nltk.download("punkt_tab")` và `punkt` ở lần đầu (lần đầu chạy backend phải có internet).
- `split()` lọc câu: bỏ câu < `MIN_SENT_LEN` (5 từ — loại tiêu đề rời) và > `MAX_SENT_LEN` (80 từ — loại đoạn dính nhau do nhận diện chấm sai).
- Mỗi câu được đánh `idx` tăng dần — chính là số `#idx` hiển thị trong frontend.

---

## 6. Bước ② — Extractive (4 thuật toán)

### 6.1. Khung chung — `extractive/base.py`

```python
class BaseExtractor:
    def extract(self, sentences):
        scores = self.score(sentences)              # do từng extractor implement
        k = max(min_keep=5, round(len(sentences) * top_k_ratio))
        ranked = sorted(range(len), key=lambda i: scores[i], reverse=True)[:k]
        ranked.sort()                               # giữ thứ tự gốc => bản tóm tắt mạch lạc
        return ExtractResult(method, picked_sentences, picked_scores)
```

Điểm mấu chốt: **chọn theo điểm cao nhất rồi sắp lại theo idx gốc** — vì câu tóm tắt mất thứ tự sẽ rất khó đọc.

`ExtractResult.as_text()` ghép các câu lại bằng dấu cách → đầu vào cho BART.

### 6.2. TF-IDF — `extractive/tfidf_extractor.py`

- `TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.95)`.
- Coi **mỗi câu là một document**, fit ngay trên doc hiện tại.
- Điểm câu = `(tổng tf-idf của câu) / (số từ câu)` — chuẩn hoá theo độ dài để không thiên vị câu dài.
- Trực giác: câu chứa nhiều từ khoá quan trọng & hiếm thì điểm cao.

### 6.3. TextRank — `extractive/textrank_extractor.py`

1. Vectorize câu bằng `TfidfVectorizer(stop_words="english")`.
2. Tính `cosine_similarity(mat)` → ma trận tương đồng N×N.
3. `np.fill_diagonal(sim, 0)` — câu không tự kết với chính mình.
4. `nx.from_numpy_array(sim)` → đồ thị có trọng số.
5. `nx.pagerank(g, alpha=0.85, max_iter=200)` → điểm centrality.
6. Trực giác: câu nào "được nhiều câu khác chỉ vào" (giống nhiều về ngữ nghĩa) → trung tâm → quan trọng. Đây là bản graph-based của PageRank.

### 6.4. K-Means — `extractive/kmeans_extractor.py`

1. Lazy load `SentenceTransformer("all-MiniLM-L6-v2")` → 384-D embedding.
2. `model.encode(texts, batch_size=32)` → ma trận embedding.
3. `KMeans(n_clusters=k, random_state=42, n_init=10)` với `k = round(N * 0.2)`.
4. Với mỗi câu, tính khoảng cách Euclid tới tâm cụm của nó → `score = 1 / (1 + d)`.
5. Trực giác: mỗi cụm = 1 chủ đề con (định nghĩa thuật ngữ, điều khoản thanh toán, điều khoản chấm dứt...). Lấy 1 câu đại diện gần tâm nhất cho mỗi cụm → tóm tắt đa chủ đề.

### 6.5. Ensemble — `extractive/ensemble.py`

- Chạy lần lượt cả 3 extractor.
- Với mỗi vector điểm: **min-max normalize** về [0,1] (để các thang điểm khác nhau cộng được).
- Weighted sum theo `[TF-IDF=1.0, TextRank=1.5, KMeans=1.0]` (TextRank được ưu tiên vì thường mạnh nhất trên văn bản pháp lý).
- Trọng số được tự chuẩn hoá về tổng = 1 trong constructor.

### 6.6. Lựa chọn động — `hybrid/pipeline.py::ExtractorFactory.build()`

Frontend gửi `extractor: "tfidf" | "textrank" | "kmeans" | "ensemble"`. Factory dựng đúng đối tượng tương ứng, truyền `TOP_K_RATIO` từ `Settings`. Ensemble dựng sẵn 3 sub-extractor.

---

## 7. Bước ③ — Abstractive (BART)

### 7.1. Chia chunk — `abstractive/chunker.py`

`LongDocChunker.chunk_by_sentences(sentences)`:

```python
for sent in sentences:
    n = tokenizer.encode(sent, add_special_tokens=False)  # đếm token CHÍNH XÁC theo BART tokenizer
    if n >= 1024:                                          # câu cực dài → cắt cứng theo token
        flush(buf); chunks.append(_hard_cut(sent))
    elif buf_tokens + n > 1024:
        flush(buf)
        buf = buf[-50:]                                    # overlap 50 câu cuối để giữ ngữ cảnh
        buf_tokens = sum(len(encode(s)) for s in buf)
    buf.append(sent); buf_tokens += n
```

Hai điểm tinh tế:
- Dùng tokenizer của BART để đếm — không phải đếm word, vì BART chia subword.
- **Overlap = 50 _câu_ cuối** (không phải 50 token). Khi văn bản đã được lọc xuống ~100–200 câu, overlap này đảm bảo câu cuối của chunk trước xuất hiện ở đầu chunk sau → BART không "quên ngữ cảnh".

### 7.2. Sinh tóm tắt — `abstractive/bart_summarizer.py`

`BartSummarizer.summarize(sentences)`:

1. `_ensure_loaded()` — lazy load `AutoTokenizer` + `AutoModelForSeq2SeqLM` từ `BART_MODEL`, đẩy lên `device`, set `eval()`. Cache vào `backend/cache/models/`.
2. `chunks = chunker.chunk_by_sentences(sentences)`.
3. Với mỗi chunk gọi `_summarize_chunk(text)`:
   ```python
   inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
   with torch.no_grad():
       ids = model.generate(
           **inputs,
           max_length=256, min_length=80,
           num_beams=4, length_penalty=2.0,
           early_stopping=True,
           no_repeat_ngram_size=3,
       )
   return tokenizer.decode(ids[0], skip_special_tokens=True)
   ```
4. Ghép các `chunk_summaries` lại. Nếu tổng vẫn > 1024 token (rất hay xảy ra với hợp đồng 47k từ) → **gọi `_summarize_chunk` lần nữa** trên chuỗi đã ghép → đây chính là vòng hierarchical.

Kết quả: `AbstractiveResult{method, text, chunks, chunk_summaries}` — frontend dùng `text` cho panel cuối, `chunk_summaries` cho phần "Xem từng chunk".

### 7.3. Vì sao chọn các tham số đó?

| Tham số | Lý do |
|---|---|
| `num_beams=4` | Cân bằng chất lượng vs tốc độ. Beam 8 đẹp hơn 5–10% nhưng chậm gấp đôi. |
| `length_penalty=2.0` | Khuyến khích câu dài hơn — hợp đồng cần thông tin đầy đủ. |
| `min_length=80` | Tránh BART "lười" trả một câu cụt. |
| `max_length=256` | Đủ cho ~150–200 từ, không vượt context decoder. |
| `no_repeat_ngram_size=3` | Tắt lặp tri-gram — vốn là bệnh kinh điển của BART. |
| `early_stopping=True` | Dừng beam khi sinh ra `</s>` chứ không kéo đến max_length. |

---

## 8. Bước ④ — Orchestrator: `hybrid/pipeline.py`

`HybridPipeline.run(raw_text, doc_id=None)`:

```python
timer = Timer()
with timer.start("clean"):    cleaned = self.cleaner.clean(raw_text)
with timer.start("split"):    sentences = self.splitter.split(cleaned)
with timer.start("extract"):  ext = self.extractor.extract(sentences)
if self.use_abstractive and ext.sentences:
    bart = self._ensure_bart()                       # lazy
    with timer.start("abstract"):
        abs_res = bart.summarize([s.text for s in ext.sentences])
return HybridResult(doc_id, raw_word_count, num_sentences, ext, abs_res, timings)
```

- `Timer` dùng `with` context để đo từng bước → ra response dưới dạng `{clean: 0.03, split: 0.12, extract: 1.8, abstract: 12.4}` (giây), tag hiển thị ở góc phải nút "Chạy pipeline lai" trên UI.
- `_ensure_bart()` chỉ load model khi thực sự cần — nhờ thế **API khởi động trong < 2 giây** dù BART nặng 1.6 GB.
- `HybridResult.to_dict()` serialize đầy đủ extractive (sentences + scores + text), abstractive (text + chunk_summaries + num_chunks), timings → JSON gửi qua REST.

---

## 9. Bước ⑤ — Evaluate

### 9.1. Tạo reference — `evaluate/reference_builder.py`

`ReferenceBuilder.get_reference(doc_id)`:
1. Lazy đọc `master_clauses.csv`. Tên file trong CSV là `*.pdf` → dùng `Path(...).stem` để khớp với `*.txt`.
2. Bỏ các cột `Filename`, `*-answer` (chỉ là Yes/No), giữ các cột clause.
3. Mỗi ô có thể là Python-list dạng string (`"['clause 1', 'clause 2']"`) → `_parse_cell()` dùng `ast.literal_eval` để parse an toàn. Nếu fail → fallback tách theo `\n;`.
4. Gộp tất cả clause của doc lại, **khử trùng lặp giữ thứ tự**, join bằng dấu cách → reference text.

### 9.2. ROUGE — `evaluate/rouge_scorer.py`

- Dùng `rouge_score.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)`.
- Trả về `RougeScore(rouge1_f, rouge2_f, rougeL_f, rouge1_p, rouge1_r)`. F1 là chính, kèm Precision/Recall của ROUGE-1 để debug (Precision thấp → BART nói lạc đề; Recall thấp → BART bỏ sót).
- `score_batch()` chạy trên list rồi lấy trung bình — dùng trong `EvalRunner`.

### 9.3. BERTScore — `evaluate/bert_scorer.py`

- Wrapper quanh `bert_score.score()` với `model_type="roberta-large"`.
- Trả về `(P, R, F)` trung bình — đánh giá theo ngữ nghĩa thay vì khớp từ. ROUGE phạt việc paraphrase; BERTScore không.

### 9.4. Benchmark — `evaluate/runner.py`

`EvalRunner.run(limit=None, doc_ids=None)`:
- Lặp từng doc_id với `tqdm`. Mỗi doc: load → build reference → chạy pipeline → tính ROUGE.
- Try/except quanh từng doc — 1 file lỗi không kill cả batch.
- Cuối cùng lưu `outputs/eval/rouge_<extractor>_<abs|ext>.json`. File này chứa `average` + `per_doc` đầy đủ. Endpoint `/api/eval/run` chỉ trả về `preview: per_doc[:5]` để không nặng response.

---

## 10. Bước ⑥ — Fine-tune BART trên CUAD (worker1, RTX 3090 24GB)

### 10.1. Build dataset — `training/dataset_builder.py`

`CuadDatasetBuilder.build_one(doc_id)`:
- `input` = câu được **TextRank** chọn (`TextRankExtractor(top_k_ratio=0.2)`).
- `target` = `ReferenceBuilder.get_reference(doc_id)` (clauses từ CSV).

`build_all(train_ratio=0.8, val_ratio=0.1)`:
- Shuffle 510 doc_id với `seed=42` (reproducible).
- Chia 80/10/10 → train ≈ 408, val ≈ 51, test ≈ 51.
- Bỏ qua doc nào input/target rỗng.

### 10.2. Trainer — `training/trainer.py`

`BartFineTuner.fit(data)`:
1. Tokenize bằng `tokenizer(input, text_target=target, max_length=1024, truncation=True)`. Sau đó truncate label thủ công về 256 (`max_target`).
2. Nếu `--use_lora`: bọc model bằng `LoraConfig(r=16, alpha=32, dropout=0.05, target_modules=["q_proj","v_proj"], task=SEQ_2_SEQ_LM)` → chỉ học < 1% tham số, tiết kiệm VRAM.
3. `Seq2SeqTrainingArguments`:
   - `epochs=3`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` → **effective batch = 16**.
   - `lr=3e-5`, `warmup_ratio=0.05`, `fp16=True` (mixed precision).
   - `predict_with_generate=True`, `generation_max_length=256` → eval bằng generate thật (đắt nhưng đúng metric).
   - `save_strategy=epoch`, `eval_strategy=epoch`, `save_total_limit=2`.
   - `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`.
4. `EarlyStoppingCallback(patience=2)` — eval_loss không giảm 2 epoch liên tiếp → dừng.
5. Compat hack: transformers ≥ 4.46 đổi tham số `tokenizer=` thành `processing_class=` → đoạn `inspect.signature(...)` chọn tên đúng.
6. Sau train: `trainer.save_model(output_dir/"final")` + save tokenizer.

### 10.3. CLI — `training/run_train.py`

```bash
python -m backend.training.run_train \
    --epochs 3 --batch_size 2 --grad_accum 8 \
    --lr 3e-5 [--use_lora] [--rebuild_dataset] \
    --output_dir outputs/bart-cuad \
    --early_stopping_patience 2
```

- Cache dataset đã build vào `outputs/eval/dataset_cuad.json` để không build lại.
- Logger ghi cả console + file `logs/train.log`.

### 10.4. Triển khai qua Fabric — `deploy/fabfile.py`

```bash
cd text_sumarisation/deploy
fab sync           # rsync code lên worker1:/home/sontn/text_sumarisation (exclude .venv, dist, outputs)
fab setup          # tạo venv, cài torch CUDA 12.1 + requirements-gpu.txt
fab gpu            # nvidia-smi + torch.cuda.is_available()
fab train --epochs=3 --batch=2 --grad-accum=8 [--lora=1]
                   # chạy nohup, ghi PID vào logs/train.pid, log vào logs/train.out
fab tail --n=200   # tail log training
fab pull-model     # rsync outputs/bart-cuad/final về local
fab serve --port=9020 / fab stop
```

Sau khi pull model về `outputs/bart-cuad/final/`, set env `BART_MODEL=/path/to/outputs/bart-cuad/final` rồi restart Flask → backend lập tức dùng checkpoint fine-tune (tokenizer/config tương thích vì cùng kiến trúc `BartForConditionalGeneration`, `d_model=1024`).

---

## 11. Backend Flask — `backend/app/`

### 11.1. Khởi tạo — `app/server.py`

`create_app()`:
- `get_settings()` + `Logger.setup()`.
- `Flask(__name__)` + `CORS(app, resources={r"/api/*": {"origins":"*"}})` — cho frontend dev gọi tự do.
- `AppState.instance()` — khởi tạo singleton sớm để fail-fast nếu DB / CSV lỗi.
- Register 5 blueprint: `summarize`, `extract`, `documents`, `eval`, `auth`.
- `/api/health` → trả `{status, device, model}` — frontend show ở hero ("device: cuda, model: facebook/bart-large-cnn").
- `/` → 302 redirect tới `/legal-ai/`.
- `/legal-ai/` và `/legal-ai/<path:path>` → serve frontend đã build (`frontend/dist/`). SPA fallback: route con không phải file → trả `index.html` để React Router xử lý.

Chạy: `PYTHONPATH=$PWD API_PORT=9020 python -m backend.app.server`.

### 11.2. State singleton — `app/state.py`

`AppState` giữ:
- `loader: ContractLoader` — list/load 510 doc.
- `references: ReferenceBuilder` — lazy đọc CSV 1 lần.
- `bart: BartSummarizer` — **dùng chung cho mọi pipeline** (không load lại model). Lazy load — chỉ tốn VRAM khi request đầu tiên có `use_abstractive=True`.
- `_pipelines: Dict[str, HybridPipeline]` — cache theo key `"<extractor>|<0/1>"`. Lần thứ hai dùng cùng cấu hình → reuse object (TF-IDF vectorizer, MiniLM model, ... không bị khởi tạo lại).

### 11.3. Các endpoint

| Endpoint | Method | Body / Query | Mô tả |
|---|---|---|---|
| `/api/health` | GET | – | `{status, device, model}` |
| `/api/documents` | GET | `?q=&page=1&page_size=50` | Filter theo substring (case-insensitive), phân trang |
| `/api/documents/<doc_id>` | GET | – | `{doc_id, word_count, text, reference}` |
| `/api/extract` | POST | `{text? | doc_id?, extractor}` | Chỉ extractive, ép `use_abstractive=False`, không cần GPU |
| `/api/summarize` | POST | `{text? | doc_id?, extractor, use_abstractive}` | Full hybrid + ROUGE nếu chạy trên doc_id |
| `/api/eval/run` | POST | `{extractor, use_abstractive, limit?, doc_ids?}` | Benchmark, ghi JSON, trả về `average + preview[5]` |
| `/api/auth/register` | POST | `{name, email, password}` | Đăng ký + issue JWT (24h) |
| `/api/auth/login` | POST | `{email, password}` | bcrypt verify + JWT |
| `/api/auth/reset-password` | POST | `{email, password}` | Reset trực tiếp (không OTP) |
| `/api/auth/forgot-password` | POST | `{email}` | Chỉ verify email tồn tại (giữ tương thích) |
| `/api/auth/me` | GET | `Authorization: Bearer <token>` | Trả profile user |
| `/api/auth/me` | DELETE | – | Soft-delete (is_deleted=true) |

`SummarizeRequest.validate()`:
- Phải có `text` hoặc `doc_id` (không cho cả hai rỗng).
- `extractor` phải thuộc `{tfidf, textrank, kmeans, ensemble}`.

`/api/summarize` luồng chi tiết:
```python
state = AppState.instance()
if req.doc_id:
    contract = state.loader.load_one(req.doc_id)
    raw_text = contract.raw_text
    reference = state.references.get_reference(req.doc_id)  # có thể rỗng

pipeline = state.get_pipeline(req.extractor, req.use_abstractive)  # cache
result = pipeline.run(raw_text, doc_id=req.doc_id)
payload_out = result.to_dict()

if reference:
    pred = result.abstractive.text if result.abstractive else result.extractive.as_text()
    payload_out["rouge"] = _rouge.score(pred, reference).to_dict()
    payload_out["reference"] = reference
return jsonify(payload_out)
```

### 11.4. Auth — `app/auth.py`

- `DSN` mặc định `postgresql://postgres:***@100.81.215.111:5432/legal_ai` — chính là PG HA cluster (worker1 primary, worker2 standby — đã ghi nhớ trong project_postgres_ha).
- `ThreadedConnectionPool(1, 8, DSN)` — pool 8 connection.
- Password: `bcrypt.hashpw(rounds=12)`.
- JWT: `HS256`, TTL 24h, secret từ env `BDP_LEGAL_JWT_SECRET` (default đã commit chỉ cho dev).
- Schema bảng `users`: `user_id, email, password_hash, full_name, avatar_url, is_active, is_deleted, deleted_at, updated_at`.
- `_user_from_request()` decode token, lookup DB, kiểm `is_active` + `is_deleted` → mọi route bảo mật đều dùng helper này.

---

## 12. Frontend — `frontend/`

### 12.1. Stack & cấu hình

- React 19.2 + Ant Design 6.3 + Tailwind 4 + Vite 8 + axios 1.16 + lucide-react.
- `vite.config.js`:
  - `base: "/legal-ai/"` → tương thích với Flask serve dưới prefix.
  - Dev: `port: 5173`, proxy `/api → http://localhost:9020` → axios chỉ cần `baseURL: "/api"`.
- `App.jsx`:
  - 5 tab: `home | search | ai-assistant | string-matching | summarization` — đồng bộ với URL hash `#/<tab>` (back/forward browser hoạt động).
  - **Bắt đăng nhập trước khi vào app**: nếu `getStoredUser()` null → render full-screen `AuthModal` không cho đóng.

### 12.2. Tab "Tóm tắt văn bản" — `components/TabSummarization.jsx`

State chính:
- `mode`: `"doc"` (chọn từ 510 file) | `"text"` (nhập / upload).
- `docList`, `docId`, `docPreview` (text + word_count + reference).
- `text` (raw).
- `extractor`, `useAbstractive`.
- `running`, `stage` (`clean|split|extract|abstract|done`), `result`.
- `serverInfo` (lấy từ `/api/health` khi mount).

Bố cục:
- **Hero**: gradient navy → purple → wine, hiện `device` + `model` từ backend.
- **Card controls** (3 cột):
  - Cột 1: `Tabs(doc | text)`. Doc: `AutoComplete` với debounce 250 ms gọi `/api/documents?q=` (top 100); chọn doc → fetch `/api/documents/<id>` show word_count + tag "có reference".
  - Cột 2: `Select` extractor + `Switch` use_abstractive.
  - Cột 3: list 4 step (Làm sạch / Tách câu / Extractive (<name>) / Abstractive (BART)). Khi chạy, dùng `setInterval(..., 700ms)` để **mô phỏng** chuyển bước hiển thị (vì backend chỉ trả về một lần ở cuối). Step "abstract" bị ẩn nếu tắt switch.
- **Nút "Chạy pipeline lai"** → gọi `summarize(...)`. Khi xong, hiện `result.timings` dưới dạng tag `clean: 0.03s`...
- **Panel kết quả** (`ResultView`):
  - 4 thẻ thống kê: số từ gốc, số câu sau split, số câu giữ lại, **tỉ lệ nén = words(abstractive || extractive) / words(raw)**.
  - Block extractive: `<ol>` từng câu với tag `#idx` + score (3 chữ số thập phân).
  - Block abstractive: paragraph chính + `<details>` "Xem từng chunk" nếu có >1 chunk.
  - Block ROUGE: `Statistic` với ROUGE-1/2/L F1 + R1 Precision/Recall (%).
- **Xuất kết quả**: build text rồi `Blob` → download `summary_<doc_id>.txt`.

### 12.3. Service layer — `services/summarization.js`

Bốc tách rõ ràng:
- `EXTRACTORS` — config dropdown.
- `listDocuments({q, page, pageSize})` → `GET /api/documents`.
- `getDocument(docId)` → `GET /api/documents/<id>`.
- `summarize({text, docId, extractor, useAbstractive})` → `POST /api/summarize`.
- `extractOnly(...)` → `POST /api/extract`.
- `health()` → `GET /api/health`.

`services/api.js` định nghĩa axios instance dùng chung với `baseURL: "/api"`.

---

## 13. Chạy local end-to-end

### 13.1. Backend (Flask)
```bash
cd /mnt/d/BigData/text_sumarisation
pip install -r deploy/requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
PYTHONPATH=$PWD API_PORT=9020 python -m backend.app.server
# → http://localhost:9020/api/health
```

### 13.2. Frontend
```bash
cd /mnt/d/BigData/text_sumarisation/frontend
npm install
npm run dev
# → http://localhost:5173/legal-ai/
```

### 13.3. Production (serve frontend qua Flask)
```bash
cd frontend && npm run build       # tạo frontend/dist với base /legal-ai/
# Flask tự serve dist khi truy cập /legal-ai/
```

### 13.4. Demo flow trong UI
1. Đăng nhập (DB legal_ai, bcrypt + JWT).
2. Vào tab "Tóm tắt văn bản".
3. Chọn 1 doc CUAD (autocomplete) → tag "có reference (CUAD)" sáng lên.
4. Chọn `TextRank` + bật Abstractive.
5. Bấm "Chạy pipeline lai" → 4 step sáng dần → kết quả hiện ra:
   - Câu được trích (top 20%) kèm điểm.
   - Bản viết lại của BART.
   - ROUGE-1/2/L vs reference.
6. Bấm "Xuất kết quả" → tải file `.txt`.

---

## 14. Hiệu năng & quan sát thực tế

| Bước | Thời gian điển hình (1 doc 7k từ, GPU 3090) |
|---|---|
| clean | < 50 ms |
| split | ~ 100 ms (NLTK punkt) |
| extract (TextRank) | 1–3 s (vector + pagerank) |
| extract (KMeans) | 3–5 s (SBERT encode chiếm phần lớn) |
| abstract (BART, ~3 chunk) | 8–15 s trên GPU, 60–90 s trên CPU |
| **Tổng** | ~10–20 s/doc |

Với doc 47k từ: BART thường chạy ~ 6 chunk + 1 vòng hierarchical → 25–40 s.

---

## 15. Slide checklist

1. **Slide 1 — Bài toán**: 510 hợp đồng dài (max 47k từ), cần tóm tắt 150–300 từ, BART chỉ chịu 1024 token.
2. **Slide 2 — Giải pháp Hybrid**: sơ đồ ở §2; nhấn mạnh "extractive lọc → abstractive viết lại".
3. **Slide 3 — Preprocess**: NFKC + regex bỏ "Page X of Y" / "Source:" + NLTK punkt + lọc câu 5–80 từ.
4. **Slide 4 — Extractive**: 4 thuật toán + trực giác. Nêu lý do TextRank thường thắng & vì sao chọn top 20%.
5. **Slide 5 — Abstractive**: BART-large-CNN, chunk 1024 token + overlap 50 câu, beam=4, length_penalty=2.0, no_repeat_ngram=3, hierarchical khi merged > 1024.
6. **Slide 6 — Reference độc đáo**: `master_clauses.csv` 83 cột clause do luật sư gán nhãn → tạo "gold summary" để chấm ROUGE — không cần tự gán nhãn.
7. **Slide 7 — Đánh giá**: ROUGE-1/2/L (P/R/F) + BERTScore roberta-large; benchmark trên 510 file → ghi `outputs/eval/rouge_*.json`.
8. **Slide 8 — Fine-tune**: input=TextRank, target=clauses; 80/10/10 split; LoRA r=16 tuỳ chọn; effective batch=16; early stopping patience=2; lưu best checkpoint theo eval_loss.
9. **Slide 9 — Triển khai**: Fabric sync → worker1 RTX 3090, fp16; backend Flask :9020, frontend React serve sau prefix `/legal-ai/`.
10. **Slide 10 — Demo live**: chọn 1 doc CUAD, nhấn Run, chỉ thanh tiến trình + timings + ROUGE.
11. **Slide 11 — Hạn chế & hướng tiếp**: BART pretrain trên tin tức → hơi gượng với legal; có thể thử LED / Pegasus-BillSum / Longformer-Encoder-Decoder; thêm chấm điểm hậu kiểm bằng QA model; mở rộng sang văn bản pháp luật tiếng Việt.

---

## 16. Phụ lục: bảng tham số tổng hợp

| Tham số | File | Mặc định | Ý nghĩa |
|---|---|---|---|
| `TOP_K_RATIO` | `config/settings.py` | 0.2 | Tỉ lệ câu extractive giữ lại |
| `MIN_SENT_LEN` | settings | 5 | Bỏ câu < N từ |
| `MAX_SENT_LEN` | settings | 80 | Bỏ câu > N từ |
| `BART_MAX_INPUT` | settings | 1024 | Token tối đa BART nhận |
| `BART_MAX_OUTPUT` | settings | 256 | Token sinh ra tối đa |
| `BART_MIN_OUTPUT` | settings | 80 | Bắt BART không trả câu quá ngắn |
| `BART_NUM_BEAMS` | settings | 4 | Beam width |
| `CHUNK_OVERLAP` | settings | 50 | Số câu chồng lấn giữa các chunk |
| `damping` | textrank_extractor | 0.85 | Hệ số PageRank |
| Ensemble weights | ensemble.py | [1.0, 1.5, 1.0] | TF-IDF / TextRank / KMeans |
| `length_penalty` | bart_summarizer | 2.0 | Khuyến khích câu dài |
| `no_repeat_ngram_size` | bart_summarizer | 3 | Chống lặp tri-gram |
| `epochs / batch / grad_accum` | run_train | 3 / 2 / 8 | Eff. batch = 16 |
| `learning_rate` | run_train | 3e-5 | Lr fine-tune BART-large |
| `lora r / alpha / dropout` | trainer | 16 / 32 / 0.05 | LoRA Q,V projections |
| `early_stopping_patience` | run_train | 2 | Epoch eval_loss không giảm thì dừng |
| `API_PORT` | settings (env) | 9020 | Flask port |
| `TOKEN_TTL_HOURS` | auth.py | 24 | JWT lifetime |
