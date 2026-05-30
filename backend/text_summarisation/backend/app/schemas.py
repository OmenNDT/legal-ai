from dataclasses import dataclass
from typing import Optional

# Schema request /api/summarize
@dataclass
class SummarizeRequest:
    text: Optional[str] = None
    doc_id: Optional[str] = None
    extractor: str = "textrank"
    use_abstractive: bool = True

    @classmethod
    def from_json(cls, data: dict) -> "SummarizeRequest":
        return cls(
            text = data.get("text"),
            doc_id = data.get("doc_id"),
            extractor = (data.get("extractor") or "textrank").lower(),
            use_abstractive = bool(data.get("use_abstractive", True))
        )

    def validate(self):
        if not self.text and not self.doc_id:
            raise ValueError("Cần truyền 'text' hoặc 'doc_id'")
        if self.extractor not in ("tfidf", "textrank", "kmeans", "ensemble"):
            raise ValueError(f"Extractor không hợp lệ: {self.extractor}")
