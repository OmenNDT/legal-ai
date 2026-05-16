from src.search.inverted_index import InvertedIndex
from src.search.bm25 import BM25
from src.search.trie import Trie, fuzzy_search

class LegalSearchEngine:
    def __init__(self):
        self.index = InvertedIndex()
        self.bm25: BM25 | None = None
        self.trie = Trie()
        self._built = False

    def add_document(self, content: str, metadata: dict | None = None) -> int:
        return self.index.add_document(content, metadata)

    def build(self):
        self.bm25 = BM25(self.index)
        for term in self.index._index:
            postings = self.index.search(term)
            total_freq = sum(p.term_frequency() for p in postings)
            self.trie.insert(term, freq=total_freq, doc_freq=len(postings))
        self._built = True

    def search(self, query: str, top_k: int = 10) -> list:
        if not self._built:
            self.build()
        return self.bm25.search(query, top_k)

    def boolean_search(self, query: str, mode: str = "and") -> list:
        terms = self.index._tokenize(query)
        if mode == "or":
            return self.index.search_or(terms)
        if mode == "phrase":
            return self.index.search_phrase(query)
        return self.index.search_and(terms)

    def autocomplete(self, prefix: str, max_results: int = 10) -> list:
        return self.trie.autocomplete(prefix, max_results)

    def fuzzy(self, word: str, max_dist: int = 2, max_results: int = 5) -> list:
        return fuzzy_search(word, list(self.index._index.keys()), max_dist, max_results)

    def explain(self, query: str, doc_id: int) -> dict:
        if not self._built:
            self.build()
        return self.bm25.explain(query, doc_id)
