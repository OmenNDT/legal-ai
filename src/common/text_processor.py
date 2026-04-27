"""Text preprocessing for Vietnamese legal text.

CRITICAL: Vietnamese text MUST be word-segmented with VnCoreNLP/RDRSegmenter
before PhoBERT tokenization. This module provides the segmentation pipeline.
"""

import re
from pathlib import Path

# VnCoreNLP will be loaded lazily to avoid startup cost
_segmenter = None


def get_segmenter(save_dir: str = "./vncorenlp"):
    """Load word segmenter for Vietnamese text.

    Tries VnCoreNLP first, falls back to underthesea if Java unavailable.
    """
    global _segmenter
    if _segmenter is not None:
        return _segmenter

    # Try VnCoreNLP first
    try:
        import py_vncorenlp
        model_dir = Path(save_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        if not (model_dir / "RDRSEG").exists():
            py_vncorenlp.download_model(save_dir=save_dir)
        _segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], save_dir=save_dir
        )
        return _segmenter
    except Exception:
        pass  # Fall through to underthesea

    # Fallback: underthesea (pure Python, no Java)
    try:
        from underthesea import word_tokenize
        _segmenter = word_tokenize
        return _segmenter
    except ImportError:
        raise ImportError(
            "No Vietnamese segmenter available. Run: pip install underthesea"
        )


def segment_text(text: str, save_dir: str = "./vncorenlp") -> str:
    """Segment Vietnamese text for PhoBERT input.

    Converts "Thông tư 99 quy định chế độ kế toán"
    → "Thông_tư 99 quy_định chế_độ kế_toán"

    Returns a single string with underscore-joined compound words.
    """
    segmenter = get_segmenter(save_dir)
    # VnCoreNLP returns list of sentences; underthesea returns list of tokens
    if callable(segmenter) and not hasattr(segmenter, 'word_segment'):
        # underthesea fallback: list of tokens
        tokens = segmenter(text)
        return " ".join(tokens)
    sentences = segmenter.word_segment(text)
    return " ".join(sentences)


def segment_texts(texts: list[str], save_dir: str = "./vncorenlp") -> list[str]:
    """Batch segment multiple texts."""
    return [segment_text(t, save_dir) for t in texts]


def clean_vietnamese(text: str) -> str:
    """Clean Vietnamese legal text."""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    # Remove excessive punctuation
    text = re.sub(r"([.!?])\1{3,}", r"\1", text)
    return text.strip()


def segment_sentences(text: str) -> list[str]:
    """Split Vietnamese text into sentences.

    Uses regex on common Vietnamese sentence boundaries.
    More robust than simple period-split for legal text.
    """
    # Split on period/newline followed by uppercase or number
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9])", text)
    # Also split on newlines that look like sentence boundaries
    sentences = []
    for part in parts:
        sub = re.split(r"\n+", part.strip())
        for s in sub:
            s = s.strip()
            if len(s) > 10:  # skip very short fragments
                sentences.append(s)
    return sentences


def extract_legal_references(text: str) -> list[dict]:
    """Extract legal document references from text.

    Returns list of dicts: [{"type": "THONG_TU", "number": "99/2015", "full": "Thông tư 99/2015/TT-BTC"}, ...]
    """
    patterns = {
        "THONG_TU": r"Thông\s+tư\s+(?:số\s+)?([\d/]+(?:/TT-\w+)?)",
        "NGHI_DINH": r"Nghị\s+định\s+(?:số\s+)?([\d/]+(?:/NĐ-CP)?)",
        "LUAT": r"Luật\s+([\w\s]+?)(?:\s+số\s+|năm|\d{4}|$)",
        "DIEU": r"Điều\s+(\d+)",
        "KHOAN": r"Khoản\s+(\d+)",
        "DIEM": r"Điểm\s+([a-zđ]+)",
    }

    results = []
    for ref_type, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            results.append({
                "type": ref_type,
                "value": match.group(1).strip() if match.lastindex else match.group(0),
                "full": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })

    # Sort by position in text
    results.sort(key=lambda x: x["start"])
    return results


def build_training_labels_for_ner(text: str, references: list[dict], tokenizer=None) -> list[str]:
    """Build BIO tags for NER training from extracted references.

    Given text and extracted legal references, produce a list of BIO tags
    aligned with tokenized text.

    Args:
        text: Original Vietnamese text
        references: Output of extract_legal_references()
        tokenizer: PhoBERT tokenizer (optional, for subword alignment)

    Returns:
        List of BIO tags (strings), one per token
    """
    # Tokenize by whitespace (will be further tokenized by PhoBERT)
    tokens = text.split()
    labels = ["O"] * len(tokens)

    # Build character-to-token mapping
    char_to_token = {}
    pos = 0
    for i, token in enumerate(tokens):
        for j in range(len(token)):
            char_to_token[pos + j] = i
        char_to_token[pos + len(token)] = i  # space position
        pos += len(token) + 1  # +1 for space

    # Assign BIO tags based on reference positions
    for ref in references:
        ref_type = ref["type"]
        start_char = ref["start"]
        end_char = ref["end"]

        # Find token indices
        start_token = char_to_token.get(start_char)
        end_token = None
        for c in range(end_char - 1, start_char, -1):
            if c in char_to_token:
                end_token = char_to_token[c]
                break

        if start_token is not None and end_token is not None:
            labels[start_token] = f"B-{ref_type}"
            for t in range(start_token + 1, end_token + 1):
                if labels[t] == "O":
                    labels[t] = f"I-{ref_type}"

    return labels