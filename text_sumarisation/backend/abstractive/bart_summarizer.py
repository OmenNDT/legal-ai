import os
import torch
import requests
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from backend.abstractive.chunker import LongDocChunker

# Kết quả tóm tắt trừu tượng
@dataclass
class AbstractiveResult:
    method: str
    text: str
    chunks: List[str]
    chunk_summaries: List[str]

# Lớp wrapper cho BART - hỗ trợ 2 chế độ:
#  - Local: tự load model và inference (CPU/GPU local)
#  - Remote: forward HTTP sang GPU worker (worker1) qua env BART_REMOTE_URL
class BartSummarizer:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: str = "cpu",
        max_input: int = 1024,
        max_output: int = 256,
        min_output: int = 80,
        num_beams: int = 4,
        cache_dir: Optional[str] = None,
        remote_url: Optional[str] = None,
        remote_timeout: float = 120.0,
        keep_chunks: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.max_input = max_input
        self.max_output = max_output
        self.min_output = min_output
        self.num_beams = num_beams
        self.cache_dir = cache_dir
        # Nếu có URL remote thì ưu tiên gọi worker1 GPU
        self.remote_url = remote_url or os.environ.get("BART_REMOTE_URL")
        self.remote_timeout = remote_timeout
        # True = ghép thẳng chunk_summaries (recall cao); False = hierarchical (ngắn hơn)
        self.keep_chunks = keep_chunks
        self._model = None
        self._tokenizer = None

    # Lazy load tokenizer local - dù chạy remote vẫn cần tokenizer để chunk
    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir = self.cache_dir)
        return self._tokenizer

    # Lazy load cả model (chỉ dùng khi không có remote)
    def _ensure_loaded(self) -> Tuple[Any, Any]:
        tokenizer = self._ensure_tokenizer()
        if self._model is None:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir = self.cache_dir)
            model.to(self.device)
            model.eval()
            self._model = model
        return tokenizer, self._model

    # Tóm tắt 1 chunk: remote ưu tiên, lỗi thì fallback local
    def _summarize_chunk(self, text: str) -> str:
        url = self.remote_url
        if url:
            try:
                return self._summarize_chunk_remote(url, text)
            except Exception as e:
                # Fallback local nếu remote chết
                print(f"[BartSummarizer] Remote thất bại ({e}), fallback local", flush = True)
        return self._summarize_chunk_local(text)

    # Gọi GPU worker qua HTTP - url đã được caller verify khác None
    def _summarize_chunk_remote(self, url: str, text: str) -> str:
        r = requests.post(
            f"{url.rstrip('/')}/summarize_chunk",
            json = {
                "text": text,
                "max_input": self.max_input,
                "max_output": self.max_output,
                "min_output": self.min_output,
                "num_beams": self.num_beams
            },
            timeout = self.remote_timeout
        )
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["summary"]

    # Inference local
    def _summarize_chunk_local(self, text: str) -> str:
        tokenizer, model = self._ensure_loaded()
        inputs = tokenizer(
            text,
            return_tensors = "pt",
            max_length = self.max_input,
            truncation = True
        ).to(self.device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_length = self.max_output,
                min_length = self.min_output,
                num_beams = self.num_beams,
                length_penalty = 2.0,
                early_stopping = True,
                no_repeat_ngram_size = 3
            )
        return tokenizer.decode(ids[0], skip_special_tokens = True).strip()

    # Tóm tắt văn bản dài: chunk -> tóm tắt từng chunk -> (giữ nguyên hoặc hierarchical)
    def summarize(self, sentences: List[str]) -> AbstractiveResult:
        tokenizer = self._ensure_tokenizer()
        chunker = LongDocChunker(tokenizer, max_tokens = self.max_input)
        chunks = chunker.chunk_by_sentences(sentences) if sentences else []
        chunk_summaries = [self._summarize_chunk(c) for c in chunks]
        merged = " ".join(chunk_summaries)
        # keep_chunks=True: trả thẳng chuỗi ghép (recall cao, dài hơn)
        # keep_chunks=False: nếu merged > max_input thì tóm tắt lại 1 lần (hierarchical, ngắn hơn)
        if self.keep_chunks:
            final = merged
        elif merged and len(tokenizer.encode(merged)) > self.max_input:
            final = self._summarize_chunk(merged)
        else:
            final = merged
        return AbstractiveResult(
            method = self.model_name,
            text = final,
            chunks = chunks,
            chunk_summaries = chunk_summaries
        )
