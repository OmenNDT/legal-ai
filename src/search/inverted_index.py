"""M4 — Inverted index for Vietnamese legal document search.

Covers: Mon Thiet ke thuat toan — Inverted index data structure.
Hash map + sorted posting lists with positional information.
Time: Build O(N*L), Term search O(k), AND search O(min posting lists)
"""

import re
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


class Posting:
    """Posting entry: doc_id + term positions."""

    __slots__ = ("doc_id", "positions")

    def __init__(self, doc_id: int, position: int):
        self.doc_id = doc_id
        self.positions = [position]

    def add_position(self, position: int):
        self.positions.append(position)

    def term_frequency(self) -> int:
        return len(self.positions)


class InvertedIndex:
    """Inverted index with positional posting lists for Vietnamese legal text."""

    def __init__(self):
        self._index: dict[str, list[Posting]] = defaultdict(list)
        self._doc_count = 0
        self._doc_lengths: dict[int, int] = {}
        self._doc_metadata: dict[int, dict] = {}

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def vocabulary_size(self) -> int:
        return len(self._index)

    @property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def add_document(self, content: str, metadata: Optional[dict] = None) -> int:
        """Index a document. Returns its doc_id."""
        doc_id = self._doc_count
        self._doc_count += 1
        if metadata:
            self._doc_metadata[doc_id] = metadata

        tokens = self._tokenize(content)
        self._doc_lengths[doc_id] = len(tokens)

        seen = {}
        for pos, token in enumerate(tokens):
            if token in seen:
                seen[token].add_position(pos)
            else:
                posting = Posting(doc_id, pos)
                self._index[token].append(posting)
                seen[token] = posting

        return doc_id

    def search(self, term: str) -> list[Posting]:
        """Single term lookup: O(k) where k = postings for term."""
        return self._index.get(term.lower(), [])

    def search_and(self, terms: list[str]) -> list[int]:
        """Boolean AND: docs containing ALL terms."""
        if not terms:
            return []
        sets = [set(p.doc_id for p in self.search(t)) for t in terms]
        result = sets[0]
        for s in sets[1:]:
            result &= s
        return sorted(result)

    def search_or(self, terms: list[str]) -> list[int]:
        """Boolean OR: docs containing ANY term."""
        ids = set()
        for t in terms:
            for p in self.search(t):
                ids.add(p.doc_id)
        return sorted(ids)

    def search_not(self, include: list[str], exclude: list[str]) -> list[int]:
        """AND NOT: docs with include terms but without exclude terms."""
        included = set(self.search_and(include))
        for t in exclude:
            for p in self.search(t):
                included.discard(p.doc_id)
        return sorted(included)

    def search_phrase(self, phrase: str) -> list[int]:
        """Phrase search: consecutive terms in order."""
        tokens = self._tokenize(phrase)
        if not tokens:
            return []
        if len(tokens) == 1:
            return [p.doc_id for p in self.search(tokens[0])]

        # Start with first term postings
        candidates = {}
        for p in self.search(tokens[0]):
            candidates[p.doc_id] = set(p.positions)

        for offset, token in enumerate(tokens[1:], 1):
            postings = self.search(token)
            new_candidates = {}
            for p in postings:
                if p.doc_id in candidates:
                    # Check for consecutive positions
                    matching = candidates[p.doc_id] & {pos - offset for pos in p.positions}
                    if matching:
                        new_candidates[p.doc_id] = {pos + offset for pos in matching}
            candidates = new_candidates
            if not candidates:
                return []

        return sorted(candidates.keys())

    def get_document_info(self, doc_id: int) -> dict:
        return self._doc_metadata.get(doc_id, {"doc_id": doc_id})

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize Vietnamese text preserving diacritics."""
        text = text.lower()
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        return [t for t in text.split() if t]

    def save(self, path: str):
        """Serialize to JSON."""
        data = {
            "doc_count": self._doc_count,
            "doc_lengths": self._doc_lengths,
            "doc_metadata": self._doc_metadata,
            "index": {
                term: [{"doc_id": p.doc_id, "positions": p.positions} for p in postings]
                for term, postings in self._index.items()
            },
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str):
        """Load from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._doc_count = data["doc_count"]
        self._doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
        self._doc_metadata = {int(k): v for k, v in data["doc_metadata"].items()}
        self._index = defaultdict(list)
        for term, postings in data["index"].items():
            for p in postings:
                posting = Posting(p["doc_id"], p["positions"][0])
                posting.positions = p["positions"]
                self._index[term].append(posting)