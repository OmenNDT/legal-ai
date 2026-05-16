from ..base import MatchResult, StringMatcher

class NaiveMatcher(StringMatcher):

    # Brute-force string matching: O(n * m).
    # Slide the pattern across the text one character at a time and compare character-by-character at each position.

    def search(self, text: str, pattern: str) -> MatchResult:
        if pattern == "":
            raise ValueError(">>> Pattern must not be empty!")

        t = self._normalize(text)
        p = self._normalize(pattern)
        n, m = len(t), len(p)

        result = MatchResult(pattern = pattern, text_length = n)
        if m > n:
            return result

        for i in range(n - m + 1):
            j = 0
            while j < m:
                result.comparisons += 1
                if t[i + j] != p[j]:
                    break
                j += 1
            if j == m:
                result.positions.append(i)

        return result
