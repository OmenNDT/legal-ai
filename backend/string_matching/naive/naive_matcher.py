from ..base import MatchResult, StringMatcher

class NaiveMatcher(StringMatcher):

    # Brute-force string matching: O(n * m).
    # Slide the pattern across the text one character at a time and compare character-by-character at each position.

    def search(self, text: str, pattern: str, trace: bool = False) -> MatchResult:
        if pattern == "":
            raise ValueError(">>> Pattern must not be empty!")

        t = self._normalize(text)
        p = self._normalize(pattern)
        n, m = len(t), len(p)

        result = MatchResult(pattern=pattern, text_length=n)

        def emit(kind, i, j, message):
            if trace:
                result.steps.append({
                    "type": kind,
                    "i": i,
                    "j": j,
                    "comparisons": result.comparisons,
                    "positions": list(result.positions),
                    "message": message
                })

        emit("info", 0, 0, f"Bắt đầu thuật toán Naive (Brute-force). Pattern dài {m}, Text dài {n}.")

        if m > n:
            emit("info", 0, 0, "Pattern rỗng hoặc dài hơn Text. Dừng.")
            return result

        for i in range(n - m + 1):
            emit("info", i, 0, f"Trượt pattern đến vị trí i={i} trong text.")
            j = 0
            while j < m:
                result.comparisons += 1
                is_match = t[i + j] == p[j]
                if is_match:
                    emit("match", i, j, f"So sánh text[{i + j}]='{text[i + j]}' với pattern[{j}]='{pattern[j]}' → khớp.")
                else:
                    emit("mismatch", i, j, f"So sánh text[{i + j}]='{text[i + j]}' với pattern[{j}]='{pattern[j]}' → không khớp. Trượt sang phải 1.")
                if not is_match:
                    break
                j += 1
            if j == m:
                result.positions.append(i)
                emit("found", i, m - 1, f"Tìm thấy pattern tại vị trí {i}.")

        emit("info", n, 0, f"Hoàn tất. Tìm thấy {len(result.positions)} vị trí với {result.comparisons} phép so sánh.")
        return result
