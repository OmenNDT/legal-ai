"""M4 — Unified search engine combining InvertedIndex + BM25 + Trie + Fuzzy."""

from src.search.inverted_index import InvertedIndex
from src.search.bm25 import BM25
from src.search.trie import Trie, levenshtein, fuzzy_search


class LegalSearchEngine:
    """End-to-end search engine for Vietnamese legal documents.

    Combines:
    - InvertedIndex for term lookup and boolean queries
    - BM25 for relevance ranking
    - Trie for autocomplete
    - Levenshtein for fuzzy matching
    """

    def __init__(self):
        self.index = InvertedIndex()
        self.bm25: BM25 | None = None
        self.trie = Trie()
        self._built = False

    def add_document(self, content: str, metadata: dict | None = None) -> int:
        """Add a legal document to the search engine."""
        doc_id = self.index.add_document(content, metadata)
        return doc_id

    def build(self):
        """Build BM25 index and Trie from the inverted index."""
        self.bm25 = BM25(self.index)
        for term in self.index._index:
            postings = self.index.search(term)
            doc_freq = len(postings)
            total_freq = sum(p.term_frequency() for p in postings)
            self.trie.insert(term, freq=total_freq, doc_freq=doc_freq)
        self._built = True

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Ranked search using BM25."""
        if not self._built:
            self.build()
        return self.bm25.search(query, top_k)

    def boolean_search(self, query: str, mode: str = "and") -> list[int]:
        """Boolean search: 'and', 'or', or phrase."""
        terms = self.index._tokenize(query)
        if mode == "or":
            return self.index.search_or(terms)
        if mode == "phrase":
            return self.index.search_phrase(query)
        return self.index.search_and(terms)

    def autocomplete(self, prefix: str, max_results: int = 10) -> list[dict]:
        """Autocomplete suggestions for a prefix."""
        return self.trie.autocomplete(prefix, max_results)

    def fuzzy(self, word: str, max_dist: int = 2, max_results: int = 5) -> list[dict]:
        """Find similar words using edit distance."""
        vocab = list(self.index._index.keys())
        return fuzzy_search(word, vocab, max_dist, max_results)

    def explain(self, query: str, doc_id: int) -> dict:
        """Explain why a document matched a query (BM25 breakdown)."""
        if not self._built:
            self.build()
        return self.bm25.explain(query, doc_id)