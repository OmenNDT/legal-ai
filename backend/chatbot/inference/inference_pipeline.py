import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from ..data_pipeline.db_loader import DbConfig
from ..data_pipeline.embedder import Embedder
from .retriever import Retriever, RetrievedChunk
from .prompt_builder import PromptBuilder
from .generator import Generator, GeneratorConfig
from .inference_logger import InferenceLogger

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

_NO_INFO = "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi này trong cơ sở dữ liệu pháp luật."

@dataclass
class InferenceResult:
    question: str
    chunks: list[RetrievedChunk]
    prompt: str
    answer: str
    latency_ms: int
    found: bool
    mode: str = "extractive"

def _build_extractive_answer(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return _NO_INFO

    # Nhóm chunks theo văn bản để trình bày gọn hơn
    from collections import defaultdict
    groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
    order: list[str] = []
    for c in chunks:
        key = c.doc_name or f"chunk_{c.chunk_id}"
        if key not in groups:
            order.append(key)
        groups[key].append(c)

    parts: list[str] = []
    for doc_name in order:
        doc_chunks = groups[doc_name]
        # Header văn bản
        parts.append(f"**{doc_name}**")
        for c in doc_chunks:
            loc_bits = []
            if c.dieu: loc_bits.append(c.dieu)
            if c.khoan: loc_bits.append(c.khoan)
            if c.diem: loc_bits.append(c.diem)
            loc = ", ".join(loc_bits)
            body = (c.full_text or "").strip()
            if loc:
                parts.append(f"{loc}: {body}")
            else:
                parts.append(body)
        parts.append("") # blank line giữa các văn bản

    # Trim trailing blank line
    while parts and parts[-1] == "":
        parts.pop()

    header = "Căn cứ các quy định pháp luật hiện hành, tôi tìm thấy những nội dung liên quan đến câu hỏi của bạn như sau:\n"
    footer = "\n\nLưu ý: Đây là trích dẫn trực tiếp từ văn bản pháp luật. Để được tư vấn chính xác cho tình huống cụ thể, bạn nên tham khảo ý kiến luật sư."
    return header + "\n".join(parts) + footer

import re

_LAW_REF_PATTERN = re.compile(
    r"(?:Nghị\s+định|Thông\s+tư|Pháp\s+lệnh|Quyết\s+định)\s+(?:số\s+)?[\w\d/\-\.]+",
    re.IGNORECASE
)

_FORBIDDEN_LAWS = re.compile(
    r"Luật\s+(?:Phòng\s+chống\s+tham\s+nhũng|Quảng\s+cáo|An\s+ninh\s+mạng|"
    r"Thuế|Giáo\s+dục|Khoáng\s+sản|Đầu\s+tư|Đấu\s+thầu|Xây\s+dựng|Nhà\s+ở|"
    r"Kinh\s+doanh\s+bất\s+động\s+sản|Tài\s+nguyên|Môi\s+trường|Du\s+lịch|"
    r"Nhập\s+cảnh|Xuất\s+cảnh|Quá\s+cảnh|Cảnh\s+sát\s+biển|Biên\s+phòng|"
    r"Phá\s+sản|Trợ\s+giúp\s+pháp\s+lý|Công\s+chứng|Trọng\s+tài|Thanh\s+tra|"
    r"Báo\s+chí|Xuất\s+bản|Sở\s+hữu\s+trí\s+tuệ|Cạnh\s+tranh|Bảo\s+vệ\s+người\s+tiêu\s+dùng|"
    r"Năng\s+lượng\s+nguyên\s+tử|Điện\s+lực|Viễn\s+thông|Công\s+nghệ\s+thông\s+tin|"
    r"Phòng\s+cháy|Chữa\s+cháy|An\s+toàn\s+thực\s+phẩm|Dược|Khám\s+chữa\s+bệnh|"
    r"Hải\s+quan|Quản\s+lý\s+thuế|Ngân\s+sách|Đầu\s+tư\s+công|Quản\s+lý\s+nợ\s+công)",
    re.IGNORECASE
)

def _is_hallucinated(answer: str, chunks: list[RetrievedChunk]) -> bool:

    if not answer or not chunks:
        return False
    # Lấy danh sách doc_name từ RAG (lowercase, không dấu phân cách)
    rag_docs = " ".join((c.doc_name or "") for c in chunks).lower()

    # 1) Nhắc Nghị định/Thông tư mà KHÔNG có Nghị định/Thông tư nào trong RAG
    if _LAW_REF_PATTERN.search(answer):
        if not _LAW_REF_PATTERN.search(rag_docs):
            return True

    # 2) Nhắc luật chuyên đề ngoài 17 luật ingest
    m = _FORBIDDEN_LAWS.search(answer)
    if m is not None:
        forbidden_match = m.group(0).lower()
        if forbidden_match not in rag_docs:
            return True
    return False

class InferencePipeline:
    def __init__(self, retriever: Retriever, builder: PromptBuilder, generator: Generator, logger: InferenceLogger | None = None, min_similarity: float = 0.62) -> None:
        self._retriever = retriever
        self._builder = builder
        self._generator = generator
        self._logger = logger
        self._min_similarity = min_similarity

    def answer(self, question: str) -> InferenceResult:
        t0 = time.monotonic()
        chunks = self._retriever.retrieve(question)
        relevant = [c for c in chunks if c.similarity >= self._min_similarity]

        if not relevant:
            ms = int((time.monotonic() - t0) * 1000)
            if self._logger:
                try: self._logger.log(question, [], _NO_INFO, ms)
                except Exception: pass
            return InferenceResult(
                question = question, chunks = [], prompt = "",
                answer = _NO_INFO, latency_ms = ms, found = False,
                mode = self._generator.mode
            )

        mode = self._generator.mode
        if mode == "lora":
            prompt = self._builder.build(question, relevant)
            answer = self._generator.generate(prompt)
            # Anti-hallucination: nếu answer nhắc tên luật/nghị định KHÔNG có trong RAG chunks
            # → fallback sang extractive (an toàn, không bịa)
            if _is_hallucinated(answer, relevant):
                prompt = ""
                answer = _build_extractive_answer(relevant)
                mode = "extractive_fallback"
        else:
            prompt = ""
            answer = _build_extractive_answer(relevant)

        ms = int((time.monotonic() - t0) * 1000)
        if self._logger:
            try: self._logger.log(question, relevant, answer, ms)
            except Exception: pass

        return InferenceResult(
            question = question, chunks = relevant, prompt = prompt,
            answer = answer, latency_ms = ms, found = True, mode = mode
        )

def _build_pipeline(no_log: bool = False) -> InferencePipeline:
    cfg = DbConfig.from_env()
    embedder = Embedder()
    logger = None if no_log else InferenceLogger(cfg)

    return InferencePipeline(
        retriever = Retriever(cfg, embedder),
        builder = PromptBuilder(),
        generator = Generator(GeneratorConfig.from_env()),
        logger = logger
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Phase 3 — Inference Pipeline")
    parser.add_argument("question", help = "Câu hỏi pháp luật")
    parser.add_argument("--no-log", action = "store_true", help = "Không ghi inference_logs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline = _build_pipeline(no_log = args.no_log)
    result = pipeline.answer(args.question)
    print("\n" + "=" * 60)
    print(f"Câu hỏi: {result.question}")
    print(f"Latency: {result.latency_ms} ms")
    print("\nChunks tìm được:")
    for c in result.chunks:
        print(f"[{c.chunk_id}] {c.dieu} {c.khoan} {c.diem} — score = {c.similarity:.3f}")
    print(f"\nTrả lời:\n{result.answer}")
    print("=" * 60)
