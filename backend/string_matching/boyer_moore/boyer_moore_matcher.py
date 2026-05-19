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

    def search(self, text: str, pattern: str, trace: bool = False) -> MatchResult:
        if pattern == "":
            raise ValueError(">>> Pattern must not be empty!")

        t = self._normalize(text)
        p = self._normalize(pattern)
        n, m = len(t), len(p)

        result = MatchResult(pattern=pattern, text_length=n)

        def emit(kind, i, j, message, **extra):
            if trace:
                step = {
                    "type": kind,
                    "i": i,
                    "j": j,
                    "comparisons": result.comparisons,
                    "positions": list(result.positions),
                    "message": message
                }
                step.update(extra)
                result.steps.append(step)

        emit("info", 0, 0, "Bắt đầu thuật toán Boyer-Moore (bad-character).")

        if m > n:
            emit("info", 0, 0, "Pattern rỗng hoặc dài hơn Text. Dừng.")
            result.extras["badChar"] = {}
            return result

        bad_char = self._build_bad_char_table(p)
        result.extras["badChar"] = bad_char
        emit("info", 0, 0, f"Bảng bad-character: {bad_char}. So sánh pattern từ PHẢI sang TRÁI.", badChar = bad_char)

        s = 0
        while s <= n - m:
            j = m - 1
            emit("info", s, j, f"Đặt pattern tại shift s={s}. Bắt đầu so sánh từ j={m - 1}.", badChar = bad_char)
            while j >= 0:
                result.comparisons += 1
                is_match = p[j] == t[s + j]
                if is_match:
                    emit("match", s, j, f"So sánh text[{s + j}]='{text[s + j]}' với pattern[{j}]='{pattern[j]}' → khớp.", badChar = bad_char)
                else:
                    emit("mismatch", s, j, f"So sánh text[{s + j}]='{text[s + j]}' với pattern[{j}]='{pattern[j]}' → không khớp.", badChar = bad_char)
                if not is_match:
                    break
                j -= 1

            if j < 0:
                result.positions.append(s)
                if s + m < n:
                    next_shift = max(1, m - bad_char.get(t[s + m], -1))
                else:
                    next_shift = 1
                emit("found", s, 0, f"Tìm thấy tại vị trí {s}. Trượt thêm {next_shift}.", badChar = bad_char)
                s += next_shift
            else:
                bc = bad_char.get(t[s + j], -1)
                shift_by = max(1, j - bc)
                emit("shift", s, j, f"Bad-char '{text[s + j]}' tại pattern index {bc}. Trượt sang phải {shift_by} bước.", badChar = bad_char, shiftBy = shift_by)
                s += shift_by

        emit("info", n, 0, f"Hoàn tất Boyer-Moore. Tìm thấy {len(result.positions)} vị trí với {result.comparisons} phép so sánh.", badChar = bad_char)
        return result
