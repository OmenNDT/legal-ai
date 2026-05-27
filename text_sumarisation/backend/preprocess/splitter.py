import re
import nltk
from dataclasses import dataclass
from typing import List
from nltk.tokenize import sent_tokenize

# Một câu sau khi tách
@dataclass
class Sentence:
    idx: int
    text: str
    word_count: int

# Lớp tách câu, ưu tiên dùng NLTK punkt, fallback regex
class SentenceSplitter:
    SENT_END_RE = re.compile(r"(?<=[\.!?])\s+(?=[A-Z0-9\"'(\[])")

    def __init__(self, min_words: int = 5, max_words: int = 80, use_nltk: bool = True):
        self.min_words = min_words
        self.max_words = max_words
        self.use_nltk = use_nltk
        self._nltk_ready = False
        if use_nltk:
            self._try_load_nltk()

    def _try_load_nltk(self):
        try:
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet = True)
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet = True)
            self._nltk_ready = True
        except Exception:
            self._nltk_ready = False

    # Tách thô bằng nltk hoặc regex
    def _raw_split(self, text: str) -> List[str]:
        if self._nltk_ready:
            return sent_tokenize(text)
        return self.SENT_END_RE.split(text)

    # Lọc câu quá ngắn hoặc quá dài
    def split(self, text: str) -> List[Sentence]:
        raw = self._raw_split(text)
        result = []
        idx = 0
        for s in raw:
            s = s.strip()
            if not s:
                continue
            wc = len(s.split())
            if wc < self.min_words or wc > self.max_words:
                continue
            result.append(Sentence(idx = idx, text = s, word_count = wc))
            idx += 1
        return result
