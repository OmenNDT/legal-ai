"""Common utilities shared across all modules."""

import json
import re
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_qa_data(path: Optional[str] = None) -> list[dict]:
    """Load Vietnamese legal Q&A data.

    Expected format: {"count": N, "columns": [...], "sample": {...}}
    or a list of {"question": ..., "context_list": ..., "qid": ..., "cid": ...}
    """
    path = Path(path) if path else DATA_DIR / "raw" / "yuiTC_sample.json"
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle wrapped format: {"count": ..., "sample": {...}}
    if isinstance(data, dict) and "sample" in data:
        return [_unwrap_qa_entry(data["sample"])]

    # Handle list format
    if isinstance(data, list):
        return [_unwrap_qa_entry(entry) for entry in data]

    return []


def _unwrap_qa_entry(entry: dict) -> dict:
    """Normalize a QA entry into consistent format."""
    return {
        "question": entry.get("question", ""),
        "context": entry.get("context_list", entry.get("context", "")),
        "qid": entry.get("qid", ""),
        "cid": entry.get("cid", ""),
    }


def load_legal_documents(path: Optional[str] = None) -> list[dict]:
    """Load processed legal documents.

    Returns list of {"id": ..., "title": ..., "content": ..., "type": ..., "metadata": ...}
    """
    path = Path(path) if path else DATA_DIR / "raw" / "uts_vlc_processed.json"
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return []


def segment_sentences(text: str) -> list[str]:
    """Simple Vietnamese sentence segmentation."""
    # Split on period, question mark, exclamation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def clean_vietnamese_text(text: str) -> str:
    """Clean Vietnamese legal text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove zero-width characters
    text = text.replace('\u200b', '').replace('\ufeff', '')
    return text.strip()


def extract_article_references(text: str) -> list[str]:
    """Extract article references from Vietnamese legal text.

    Patterns: 'Điều X', 'Khoản X', 'Thông tư X', 'Nghị định X', etc.
    """
    patterns = [
        r'Điều\s+\d+',
        r'Khoản\s+\d+',
        r'Thông tư\s+[\d/]+',
        r'Nghị định\s+[\d/]+',
        r'Luật\s+[\w\s]+',
    ]
    refs = []
    for pattern in patterns:
        refs.extend(re.findall(pattern, text))
    return list(set(refs))