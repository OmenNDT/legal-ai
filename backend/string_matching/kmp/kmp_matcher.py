from typing import List
from ..base import MatchResult, StringMatcher

class KMPMatcher(StringMatcher):

    # Knuth-Morris-Pratt: O(n + m).
    # Precompute the LPS (Longest Proper Prefix that is also Suffix) array for the pattern, then scan the text without ever moving the text cursor backwards.

    @staticmethod
    def _build_lps(p: str) -> List[int]:
        m = len(p)
        lps = [0] * m
        length = 0
        i = 1
        while i < m:
            if p[i] == p[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    def search(self, text: str, pattern: str) -> MatchResult:
        if pattern == "":
            raise ValueError(">>> Pattern must not be empty!")

        t = self._normalize(text)
        p = self._normalize(pattern)
        n, m = len(t), len(p)

        result = MatchResult(pattern=pattern, text_length=n)
        if m > n:
            return result

        lps = self._build_lps(p)
        i = j = 0
        while i < n:
            result.comparisons += 1
            if t[i] == p[j]:
                i += 1
                j += 1
                if j == m:
                    result.positions.append(i - j)
                    j = lps[j - 1]
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1

        return result