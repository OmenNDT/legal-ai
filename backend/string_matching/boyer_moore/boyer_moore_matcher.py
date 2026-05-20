from typing import Dict, List
from ..base import MatchResult, StringMatcher

class BoyerMooreMatcher(StringMatcher):

    # Boyer-Moore with both bad-character and good-suffix heuristics.
    # Compare the pattern against the text from RIGHT to LEFT.
    # On a mismatch, shift the pattern using the MAXIMUM of:
    #   1. bad-character shift (last-occurrence table)
    #   2. good-suffix shift (matched suffix reoccurrence)
    # This yields larger jumps and fewer comparisons than either heuristic alone.

    @staticmethod
    def _build_bad_char_table(p: str) -> Dict[str, int]:
        table: Dict[str, int] = {}
        for i, ch in enumerate(p):
            table[ch] = i
        return table

    @staticmethod
    def _build_good_suffix_table(p: str) -> List[int]:
        """Build the good-suffix shift table.

        good_suffix[j] = shift amount when mismatch occurs at pattern position j,
        meaning p[j+1..m-1] matched the text but p[j] did not.
        """
        m = len(p)
        # suffix[i] = length of longest suffix of p[0..i] that is also a suffix of p
        suffix = [0] * m
        suffix[m - 1] = m
        g = m - 1
        for i in range(m - 2, -1, -1):
            if i > g and suffix[i + m - 1 - f] < i - g:
                suffix[i] = suffix[i + m - 1 - f]
            else:
                if i < g:
                    g = i
                f = i
                while g >= 0 and p[g] == p[g + m - 1 - f]:
                    g -= 1
                suffix[i] = f - g

        # good_suffix[j] = shift when mismatch at position j
        good_suffix = [m] * m
        j = 0
        for i in range(m - 1, -1, -1):
            if suffix[i] == i + 1:
                while j < m - 1 - i:
                    if good_suffix[j] == m:
                        good_suffix[j] = m - 1 - i
                    j += 1
        for i in range(m - 1):
            good_suffix[m - 1 - suffix[i]] = m - 1 - i

        return good_suffix

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

        emit("info", 0, 0, "Bắt đầu thuật toán Boyer-Moore (bad-character + good-suffix).")

        if m > n:
            emit("info", 0, 0, "Pattern rỗng hoặc dài hơn Text. Dừng.")
            result.extras["badChar"] = {}
            result.extras["goodSuffix"] = []
            return result

        bad_char = self._build_bad_char_table(p)
        good_suffix = self._build_good_suffix_table(p)
        result.extras["badChar"] = bad_char
        result.extras["goodSuffix"] = good_suffix
        emit("info", 0, 0, f"Bảng bad-character: {bad_char}. Bảng good-suffix: {good_suffix}. So sánh pattern từ PHẢI sang TRÁI.", badChar=bad_char, goodSuffix=good_suffix)

        s = 0
        while s <= n - m:
            j = m - 1
            emit("info", s, j, f"Đặt pattern tại shift s={s}. Bắt đầu so sánh từ j={m - 1}.", badChar=bad_char, goodSuffix=good_suffix)
            while j >= 0:
                result.comparisons += 1
                is_match = p[j] == t[s + j]
                if is_match:
                    emit("match", s, j, f"So sánh text[{s + j}]='{text[s + j]}' với pattern[{j}]='{pattern[j]}' → khớp.", badChar=bad_char, goodSuffix=good_suffix)
                else:
                    emit("mismatch", s, j, f"So sánh text[{s + j}]='{text[s + j]}' với pattern[{j}]='{pattern[j]}' → không khớp.", badChar=bad_char, goodSuffix=good_suffix)
                if not is_match:
                    break
                j -= 1

            if j < 0:
                result.positions.append(s)
                # After full match, use good-suffix rule to find next occurrence.
                # good_suffix[0] tells us how much to shift when the entire pattern matched.
                next_shift = good_suffix[0] if good_suffix[0] < m else m
                emit("found", s, 0, f"Tìm thấy tại vị trí {s}. Trượt thêm {next_shift} theo good-suffix rule.", badChar=bad_char, goodSuffix=good_suffix, shiftBy=next_shift)
                s += next_shift
            else:
                # Case 1: mismatch with no partial match (j == m-1) -> only bad-char applies
                # Case 2: mismatch after partial match (j < m-1) -> apply BOTH rules, take MAX
                bc = bad_char.get(t[s + j], -1)
                bc_shift = max(1, j - bc)

                if j == m - 1:
                    # No partial match — only bad-character rule applies
                    shift_by = bc_shift
                    emit("shift", s, j, f"Không khớp ngay tại j={j}. Bad-char '{text[s + j]}' tại pattern index {bc}. Trượt sang phải {shift_by} bước (chỉ bad-character).", badChar=bad_char, goodSuffix=good_suffix, bcShift=bc_shift, gsShift=0, shiftBy=shift_by)
                else:
                    # Partial match — apply BOTH rules, take MAX
                    gs_shift = good_suffix[j]
                    shift_by = max(bc_shift, gs_shift)
                    rule_used = "bad-character" if bc_shift >= gs_shift else "good-suffix"
                    emit("shift", s, j, f"Khớp 1 phần (suffix từ j+1 đã khớp). Bad-char shift={bc_shift}, Good-suffix shift={gs_shift}. Trượt sang phải {shift_by} bước (theo {rule_used}).", badChar=bad_char, goodSuffix=good_suffix, bcShift=bc_shift, gsShift=gs_shift, shiftBy=shift_by)

                s += shift_by

        emit("info", n, 0, f"Hoàn tất Boyer-Moore. Tìm thấy {len(result.positions)} vị trí với {result.comparisons} phép so sánh.", badChar=bad_char, goodSuffix=good_suffix)
        return result
