from typing import Dict
from ..base import MatchResult, StringMatcher

class BoyerMooreMatcher(StringMatcher):

    # Boyer-Moore with the bad-character heuristic. 
    # Compare the pattern against the text from RIGHT to LEFT. On a mismatch, shift the pattern using the last-occurrence table of the mismatching text character, allowing large jumps.

    @staticmethod
    def _build_bad_char_table(p: str) -> Dict[str, int]:
        table: Dict[str, int] = {}
        for i, ch in enumerate(p):
            table[ch] = i
        return table

    def search(self, text: str, pattern: str) -> MatchResult:
        if pattern == "":
            raise ValueError(">>> Pattern must not be empty!")

        t = self._normalize(text)
        p = self._normalize(pattern)
        n, m = len(t), len(p)

        result = MatchResult(pattern=pattern, text_length=n)
        if m > n:
            return result

        bad_char = self._build_bad_char_table(p)

        s = 0  # Shift of the pattern relative to text.
        while s <= n - m:
            j = m - 1
            while j >= 0:
                result.comparisons += 1
                if p[j] != t[s + j]:
                    break
                j -= 1

            if j < 0:
                result.positions.append(s)
                # Shift by m if at end, else align with char after match.
                if s + m < n:
                    s += m - bad_char.get(t[s + m], -1)
                else:
                    s += 1
            else:
                bc = bad_char.get(t[s + j], -1)
                s += max(1, j - bc)

        return result
