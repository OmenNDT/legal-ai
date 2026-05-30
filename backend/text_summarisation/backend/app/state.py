from typing import Dict, Optional
from backend.config.settings import Settings, get_settings
from backend.preprocess.loader import ContractLoader
from backend.evaluate.reference_builder import ReferenceBuilder
from backend.abstractive.bart_summarizer import BartSummarizer
from backend.extractive.label_guided import LabelGuidedReranker
from backend.hybrid.pipeline import HybridPipeline

# Singleton giữ các đối tượng nặng (model, dataset) trong suốt vòng đời Flask
class AppState:
    _instance: Optional["AppState"] = None

    def __init__(self):
        self.settings: Settings = get_settings()
        self.loader = ContractLoader(self.settings.TXT_DIR)
        self.references = ReferenceBuilder(self.settings.CSV_FILE)
        # BART dùng chung - lười load
        self.bart = BartSummarizer(
            model_name = self.settings.BART_MODEL,
            device = self.settings.device(),
            max_input = self.settings.BART_MAX_INPUT,
            max_output = self.settings.BART_MAX_OUTPUT,
            min_output = self.settings.BART_MIN_OUTPUT,
            num_beams = self.settings.BART_NUM_BEAMS,
            cache_dir = str(self.settings.MODEL_CACHE),
            keep_chunks = self.settings.BART_KEEP_CHUNKS
        )
        # Reranker dùng chung (lười encode SBERT, chỉ dùng khi request có doc_id)
        self.reranker: Optional[LabelGuidedReranker] = None
        if self.settings.LABEL_GUIDED:
            self.reranker = LabelGuidedReranker(
                reference_builder = self.references,
                sbert_model = self.settings.SBERT_MODEL,
                device = self.settings.device(),
                sim_threshold = self.settings.LABEL_SIM_THRESHOLD,
                max_extra_ratio = self.settings.LABEL_MAX_EXTRA_RATIO
            )
        # Cache pipeline theo cấu hình để không khởi tạo lại
        self._pipelines: Dict[str, HybridPipeline] = {}

    @classmethod
    def instance(cls) -> "AppState":
        if cls._instance is None:
            cls._instance = AppState()
        return cls._instance

    # Lấy pipeline tương ứng (extractor, use_abstractive)
    def get_pipeline(self, extractor: str, use_abstractive: bool) -> HybridPipeline:
        key = f"{extractor}|{int(use_abstractive)}"
        if key not in self._pipelines:
            self._pipelines[key] = HybridPipeline(
                settings = self.settings,
                extractor_name = extractor,
                use_abstractive = use_abstractive,
                bart = self.bart,
                reranker = self.reranker
            )
        return self._pipelines[key]