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

class InferencePipeline:
    def __init__(self, retriever: Retriever, builder: PromptBuilder, generator: Generator, logger: InferenceLogger | None = None, min_similarity: float = 0.35) -> None:
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
                self._logger.log(question, [], _NO_INFO, ms)
            return InferenceResult(
                question = question,
                chunks = [],
                prompt = "",
                answer = _NO_INFO,
                latency_ms = ms,
                found = False
            )

        prompt = self._builder.build(question, relevant)
        answer = self._generator.generate(prompt)
        ms = int((time.monotonic() - t0) * 1000)

        if self._logger:
            self._logger.log(question, relevant, answer, ms)

        return InferenceResult(
            question = question,
            chunks = relevant,
            prompt = prompt,
            answer = answer,
            latency_ms = ms,
            found = True
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
        print(f"  [{c.chunk_id}] {c.dieu} {c.khoan} {c.diem} — score={c.similarity:.3f}")
    print(f"\nTrả lời:\n{result.answer}")
    print("=" * 60)
