"""Real LoRA Preprocessor for Phần 1 (Preprocessing).

Thay thế MockPreprocessor bằng model LoRA đã train để thực hiện
intent classification và NER thật trên câu hỏi người dùng.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.common.config import (
    LORA_CHECKPOINT_PATH,
    PHOBERT_MODEL,
    INTENT_LABELS,
    NER_LABELS,
)
from src.rag_pipeline.contracts import ProcessedQuestion, ExtractedEntity

logger = logging.getLogger(__name__)

# ── Optional dependencies ───────────────────────────────

try:
    from src.chatbot.intent_classifier import PhoBERTIntentClassifier
    from src.chatbot.ner_tagger import PhoBERTNERTagger

    _MODELS_AVAILABLE = True
except ImportError as exc:
    logger.warning("Intent/NER models unavailable: %s", exc)
    _MODELS_AVAILABLE = False

try:
    import peft  # noqa: F401

    _PEFT_AVAILABLE = True
except ImportError:
    logger.warning("peft not installed, LoRA models may not load correctly")
    _PEFT_AVAILABLE = False

try:
    from underthesea import word_tokenize

    _WORD_SEGMENT_AVAILABLE = True
except ImportError:
    _WORD_SEGMENT_AVAILABLE = False


# ── LoRA Preprocessor ───────────────────────────────────

class LoRAPreprocessor:
    """Load LoRA checkpoint và thực hiện tiền xử lý câu hỏi thật."""

    def __init__(self, checkpoint_path=None, device="cpu"):
        self.checkpoint_path = checkpoint_path or str(LORA_CHECKPOINT_PATH)
        self.device = device
        self._loaded = False
        self._fallback = False
        self._tokenizer = None
        self._intent_model = None
        self._ner_model = None
        self._intent_id2label = None
        self._ner_id2label = None

    def _load(self) -> None:
        """Lazy-load model trên lần gọi process() đầu tiên."""
        if self._loaded:
            return

        if not _MODELS_AVAILABLE or not _PEFT_AVAILABLE:
            logger.warning(
                "Missing dependencies (models=%s, peft=%s), falling back to basic preprocessing",
                _MODELS_AVAILABLE,
                _PEFT_AVAILABLE,
            )
            self._fallback = True
            self._loaded = True
            return

        if not Path(self.checkpoint_path).exists():
            logger.warning(
                "Checkpoint not found at %s, falling back to basic preprocessing",
                self.checkpoint_path,
            )
            self._fallback = True
            self._loaded = True
            return

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state = checkpoint.get("model_state_dict", checkpoint)

            self._tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

            # --- Intent classifier ---
            intent_model = PhoBERTIntentClassifier()
            intent_state = {
                k.replace("intent_classifier.", ""): v
                for k, v in state.items()
                if k.startswith("intent_classifier.")
            }
            phobert_state = {
                k.replace("phobert.", ""): v
                for k, v in state.items()
                if k.startswith("phobert.")
            }
            intent_model.phobert.load_state_dict(phobert_state, strict=False)
            intent_model.classifier.load_state_dict(intent_state, strict=False)
            intent_model.to(self.device)
            intent_model.eval()
            self._intent_model = intent_model

            # --- NER tagger ---
            ner_model = PhoBERTNERTagger()
            ner_state = {
                k.replace("ner_classifier.", ""): v
                for k, v in state.items()
                if k.startswith("ner_classifier.")
            }
            ner_model.phobert.load_state_dict(phobert_state, strict=False)
            ner_model.classifier.load_state_dict(ner_state, strict=False)
            ner_model.to(self.device)
            ner_model.eval()
            self._ner_model = ner_model

            # --- Label mappings ---
            intent_label2id = checkpoint.get(
                "intent_label2id",
                {label: i for i, label in enumerate(INTENT_LABELS)},
            )
            self._intent_id2label = {v: k for k, v in intent_label2id.items()}
            intent_model.id2label = self._intent_id2label
            intent_model.label2id = intent_label2id

            ner_label2id = checkpoint.get(
                "ner_label2id",
                {label: i for i, label in enumerate(NER_LABELS)},
            )
            self._ner_id2label = {v: k for k, v in ner_label2id.items()}
            ner_model.id2label = self._ner_id2label
            ner_model.label2id = ner_label2id

            self._loaded = True
            logger.info("Loaded LoRA checkpoint from %s", self.checkpoint_path)
        except Exception as exc:
            logger.warning(
                "Failed to load LoRA checkpoint: %s, falling back to basic preprocessing",
                exc,
            )
            self._fallback = True
            self._loaded = True

    def process(self, raw_question: str) -> ProcessedQuestion:
        """Tiền xử lý câu hỏi và trả về ProcessedQuestion."""
        self._load()

        # Word segmentation
        if _WORD_SEGMENT_AVAILABLE:
            segmented = " ".join(word_tokenize(raw_question))
        else:
            segmented = raw_question

        # Fallback: trả về dữ liệu cơ bản nếu model không load được
        if self._fallback or self._intent_model is None or self._ner_model is None:
            return ProcessedQuestion(
                raw_text=raw_question,
                segmented_text=segmented,
                intent="unknown",
                intent_confidence=0.0,
                entities=[],
                filters={},
            )

        # Intent classification
        intent_result = self._intent_model.predict(
            raw_question, self._tokenizer, self.device
        )
        intent = intent_result.get("intent", "unknown")
        confidence = float(intent_result.get("confidence", 0.0))

        # NER extraction
        ner_results = self._ner_model.predict(
            raw_question, self._tokenizer, self.device
        )
        entities = [
            ExtractedEntity(
                text=e["text"],
                label=e["entity"],
                start=e["start"],
                end=e["end"],
            )
            for e in ner_results
        ]

        filters = self._build_filters(entities)

        return ProcessedQuestion(
            raw_text=raw_question,
            segmented_text=segmented,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            filters=filters,
        )

    @staticmethod
    def _build_filters(entities: list[ExtractedEntity]) -> dict:
        """Build metadata filters từ entities."""
        filters = {}
        for ent in entities:
            if ent.label == "LUAT":
                filters["doc_type"] = "law"
            elif ent.label == "NGHI_DINH":
                filters["doc_type"] = "decree"
            elif ent.label == "THONG_TU":
                filters["doc_type"] = "circular"
        return filters
