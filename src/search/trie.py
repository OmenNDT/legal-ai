"""M4 — Trie for autocomplete + Levenshtein fuzzy search.

Covers: Mon Thiet ke thuat toan — Trie (prefix tree) + edit distance.
Time: Insert O(m), Prefix search O(m), Autocomplete O(m+k)
Levenshtein: O(m*n) DP
"""

from typing import Optional


class TrieNode:
    __slots__ = ("children", "is_end", "freq", "doc_freq")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.freq: int = 0
        self.doc_freq: int = 0


class Trie:
    """Trie for Vietnamese legal term autocomplete."""

    def __init__(self):
        self.root = TrieNode()
        self._total = 0

    def insert(self, word: str, freq: int = 1, doc_freq: int = 0):
        """Insert word with corpus frequency. O(m) where m=len(word)."""
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if not node.is_end:
            self._total += 1
        node.is_end = True
        node.freq += freq
        node.doc_freq += doc_freq

    def search_exact(self, word: str) -> Optional[TrieNode]:
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node if node.is_end else None

    def autocomplete(self, prefix: str, max_results: int = 10) -> list[dict]:
        """Prefix completion. O(m + k) where m=prefix len, k=results."""
        node = self.root
        for ch in prefix.lower():
            if ch not in node.children:
                return []
            node = node.children[ch]

        results = []
        self._collect(node, prefix.lower(), results)
        results.sort(key=lambda x: x["freq"], reverse=True)
        return results[:max_results]

    def _collect(self, node: TrieNode, current: str, results: list):
        if node.is_end:
            results.append({"word": current, "freq": node.freq, "doc_freq": node.doc_freq})
        for ch, child in node.children.items():
            self._collect(child, current + ch, results)


def levenshtein(s1: str, s2: str, max_dist: Optional[int] = None) -> int:
    """Levenshtein edit distance. O(m*n) DP.

    Args:
        s1, s2: strings to compare
        max_dist: early termination threshold (None = no limit)
    """
    m, n = len(s1), len(s2)
    if max_dist is not None and abs(m - n) > max_dist:
        return max_dist + 1

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        row_min = curr[0]
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            row_min = min(row_min, curr[j])
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev, curr = curr, prev

    return prev[n]


def fuzzy_search(word: str, vocabulary: list[str], max_dist: int = 2, max_results: int = 5) -> list[dict]:
    """Find closest vocabulary words within edit distance.

    Time: O(V * m * n) where V = vocab size, m,n = string lengths
    For production: use BK-tree or Levenshtein automaton for O(log V * m * n)
    """
    results = []
    for v in vocabulary:
        d = levenshtein(word.lower(), v.lower(), max_dist=max_dist)
        if d <= max_dist:
            results.append({"word": v, "distance": d})
    results.sort(key=lambda x: x["distance"])
    return results[:max_results]