import re
from pathlib import Path

_segmenter = None


def get_segmenter(save_dir: str = "./vncorenlp"):
    global _segmenter
    if _segmenter is not None:
        return _segmenter
    try:
        import py_vncorenlp
        model_dir = Path(save_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        if not (model_dir / "RDRSEG").exists():
            py_vncorenlp.download_model(save_dir=save_dir)
        _segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
        return _segmenter
    except Exception:
        pass
    try:
        from underthesea import word_tokenize
        _segmenter = word_tokenize
        return _segmenter
    except ImportError:
        raise ImportError("No Vietnamese segmenter available. Run: pip install underthesea")


def segment_text(text: str, save_dir: str = "./vncorenlp") -> str:
    segmenter = get_segmenter(save_dir)
    if callable(segmenter) and not hasattr(segmenter, "word_segment"):
        return " ".join(segmenter(text))
    return " ".join(segmenter.word_segment(text))


def segment_texts(texts: list, save_dir: str = "./vncorenlp") -> list:
    return [segment_text(t, save_dir) for t in texts]


def clean_vietnamese(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("​", "").replace("﻿", "")
    text = re.sub(r"([.!?])\1{3,}", r"\1", text)
    return text.strip()


def segment_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ0-9])", text)
    sentences = []
    for part in parts:
        for s in re.split(r"\n+", part.strip()):
            s = s.strip()
            if len(s) > 10:
                sentences.append(s)
    return sentences


def extract_legal_references(text: str) -> list:
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
    results.sort(key=lambda x: x["start"])
    return results


def build_training_labels_for_ner(text: str, references: list, tokenizer=None) -> list:
    tokens = text.split()
    labels = ["O"] * len(tokens)
    char_to_token = {}
    pos = 0
    for i, token in enumerate(tokens):
        for j in range(len(token)):
            char_to_token[pos + j] = i
        char_to_token[pos + len(token)] = i
        pos += len(token) + 1
    for ref in references:
        start_token = char_to_token.get(ref["start"])
        end_token = None
        for c in range(ref["end"] - 1, ref["start"], -1):
            if c in char_to_token:
                end_token = char_to_token[c]
                break
        if start_token is not None and end_token is not None:
            labels[start_token] = f"B-{ref['type']}"
            for t in range(start_token + 1, end_token + 1):
                if labels[t] == "O":
                    labels[t] = f"I-{ref['type']}"
    return labels
