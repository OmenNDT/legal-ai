import logging
import re
from dataclasses import asdict

logger = logging.getLogger(__name__)

from backend.common.text_processor import clean_vietnamese, segment_text

class TextCleaner:
    _HEADER_FOOTER_RE = re.compile(
        r"(?:^|\n)\s*(?:trang\s+\d+|\d+\s*/\s*\d+|www\.\S+|https?://\S+)\s*(?:\n|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    _PAGE_NUMBER_RE = re.compile(r"\n\s*\d{1,4}\s*\n")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def __init__(self, spark=None, segmenter_dir: str = "./vncorenlp"):
        self._spark = spark
        self._segmenter_dir = segmenter_dir

    def clean(self, documents: list) -> list:
        logger.info("[Clean] Input: %d doc(s)", len(documents))
        if self._spark is not None:
            result = self._clean_spark(documents)
        else:
            result = [self._clean_doc(doc) for doc in documents]
        logger.info("[Clean] Done: %d doc(s)", len(result))
        return result

    def _clean_spark(self, documents: list) -> list:
        segmenter_dir = self._segmenter_dir
        rdd = self._spark.sparkContext.parallelize(documents)

        def _clean_worker(doc):
            import re
            from backend.common.text_processor import clean_vietnamese, segment_text

            HEADER_FOOTER_RE = re.compile(
                r"(?:^|\n)\s*(?:trang\s+\d+|\d+\s*/\s*\d+|www\.\S+|https?://\S+)\s*(?:\n|$)",
                re.IGNORECASE | re.MULTILINE,
            )
            PAGE_NUMBER_RE = re.compile(r"\n\s*\d{1,4}\s*\n")
            MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
            CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

            from dataclasses import asdict
            raw = asdict(doc) if hasattr(doc, "__dataclass_fields__") else dict(doc)
            text = raw.get("raw_text", "")
            text = CONTROL_CHARS_RE.sub("", text)
            text = HEADER_FOOTER_RE.sub("\n", text)
            text = PAGE_NUMBER_RE.sub("\n", text)
            text = MULTI_NEWLINE_RE.sub("\n\n", text)
            text = clean_vietnamese(text)
            try:
                segmented = segment_text(text, segmenter_dir)
            except Exception:
                segmented = text
            return {**raw, "clean_text": text, "segmented_text": segmented}

        return rdd.map(_clean_worker).collect()

    def _clean_doc(self, doc) -> dict:
        raw = asdict(doc) if hasattr(doc, "__dataclass_fields__") else dict(doc)
        text = raw.get("raw_text", "")
        text = self._CONTROL_CHARS_RE.sub("", text)
        text = self._HEADER_FOOTER_RE.sub("\n", text)
        text = self._PAGE_NUMBER_RE.sub("\n", text)
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)
        text = clean_vietnamese(text)
        try:
            segmented = segment_text(text, self._segmenter_dir)
        except Exception:
            segmented = text
        return {**raw, "clean_text": text, "segmented_text": segmented}
