from dataclasses import dataclass, field
from typing import Optional, List
from backend.config.settings import Settings
from backend.preprocess.cleaner import TextCleaner
from backend.preprocess.splitter import SentenceSplitter, Sentence
from backend.extractive.base import BaseExtractor, ExtractResult
from backend.extractive.tfidf_extractor import TfidfExtractor
from backend.extractive.textrank_extractor import TextRankExtractor
from backend.extractive.kmeans_extractor import KMeansExtractor
from backend.extractive.ensemble import EnsembleExtractor
from backend.abstractive.bart_summarizer import BartSummarizer, AbstractiveResult
from backend.extractive.label_guided import LabelGuidedReranker
from backend.evaluate.reference_builder import ReferenceBuilder
from backend.utils.timer import Timer

# Output đầy đủ của pipeline lai
@dataclass
class HybridResult:
    doc_id: Optional[str]
    raw_word_count: int
    num_sentences: int
    extractive: ExtractResult
    abstractive: Optional[AbstractiveResult]
    timings: dict = field(default_factory = dict)

    # Dict hoá để gửi qua API
    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "raw_word_count": self.raw_word_count,
            "num_sentences": self.num_sentences,
            "extractive": {
                "method": self.extractive.method,
                "sentences": [{"idx": s.idx, "text": s.text, "words": s.word_count} for s in self.extractive.sentences],
                "scores": self.extractive.scores,
                "text": self.extractive.as_text()
            },
            "abstractive": None if self.abstractive is None else {
                "method": self.abstractive.method,
                "text": self.abstractive.text,
                "num_chunks": len(self.abstractive.chunks),
                "chunk_summaries": self.abstractive.chunk_summaries
            },
            "timings": self.timings
        }

# Tạo extractor theo tên
class ExtractorFactory:
    @staticmethod
    def build(name: str, settings: Settings) -> BaseExtractor:
        name = (name or "textrank").lower()
        if name == "tfidf":
            return TfidfExtractor(top_k_ratio = settings.TOP_K_RATIO)
        if name == "textrank":
            return TextRankExtractor(top_k_ratio = settings.TOP_K_RATIO)
        if name == "kmeans":
            return KMeansExtractor(top_k_ratio = settings.TOP_K_RATIO, sbert_model = settings.SBERT_MODEL, device = settings.device())
        if name == "ensemble":
            return EnsembleExtractor(
                extractors = [
                    TfidfExtractor(top_k_ratio = settings.TOP_K_RATIO),
                    TextRankExtractor(top_k_ratio = settings.TOP_K_RATIO),
                    KMeansExtractor(top_k_ratio = settings.TOP_K_RATIO, sbert_model = settings.SBERT_MODEL, device = settings.device()),
                ],
                weights = [1.0, 1.5, 1.0],
                top_k_ratio = settings.TOP_K_RATIO
            )
        raise ValueError(f"Extractor không hợp lệ: {name}")

# Pipeline điều phối: clean -> split -> extractive -> (abstractive)
class HybridPipeline:
    def __init__(self, settings: Settings, extractor_name: str = "textrank", use_abstractive: bool = True, bart: Optional[BartSummarizer] = None, reranker: Optional[LabelGuidedReranker] = None):
        self.settings = settings
        self.cleaner = TextCleaner()
        self.splitter = SentenceSplitter(min_words = settings.MIN_SENT_LEN, max_words = settings.MAX_SENT_LEN)
        self.extractor = ExtractorFactory.build(extractor_name, settings)
        self.use_abstractive = use_abstractive
        # Cho phép truyền BART có sẵn để chia sẻ giữa nhiều request
        self.bart = bart
        # Reranker dùng nhãn (chỉ active khi LABEL_GUIDED=True và request có doc_id)
        self.reranker = reranker

    # Lười khởi tạo BART để Flask khởi động không cần tải model
    def _ensure_bart(self) -> BartSummarizer:
        if self.bart is None:
            self.bart = BartSummarizer(
                model_name = self.settings.BART_MODEL,
                device = self.settings.device(),
                max_input = self.settings.BART_MAX_INPUT,
                max_output = self.settings.BART_MAX_OUTPUT,
                min_output = self.settings.BART_MIN_OUTPUT,
                num_beams = self.settings.BART_NUM_BEAMS,
                cache_dir = str(self.settings.MODEL_CACHE)
            )
        return self.bart

    # Chạy đầy đủ trên một văn bản thô
    def run(self, raw_text: str, doc_id: Optional[str] = None) -> HybridResult:
        timer = Timer()
        # Bước làm sạch
        with timer.start("clean"):
            cleaned = self.cleaner.clean(raw_text)
        timer.stop()
        # Bước tách câu
        with timer.start("split"):
            sentences: List[Sentence] = self.splitter.split(cleaned)
        timer.stop()
        # Bước trích xuất
        with timer.start("extract"):
            ext = self.extractor.extract(sentences)
        timer.stop()
        # Bước rerank theo nhãn (nếu có doc_id và reranker được bật)
        if self.reranker is not None and doc_id:
            with timer.start("label_rerank"):
                ext = self.reranker.rerank(doc_id, sentences, ext)
            timer.stop()
        # Bước trừu tượng (tùy chọn)
        abs_res = None
        if self.use_abstractive and ext.sentences:
            bart = self._ensure_bart()
            with timer.start("abstract"):
                abs_res = bart.summarize([s.text for s in ext.sentences])
            timer.stop()
        return HybridResult(
            doc_id = doc_id,
            raw_word_count = len(cleaned.split()),
            num_sentences = len(sentences),
            extractive = ext,
            abstractive = abs_res,
            timings = timer.report()
        )