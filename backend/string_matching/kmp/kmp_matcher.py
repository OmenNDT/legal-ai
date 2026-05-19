from typing import List, Tuple, Dict, Any
from ..base import MatchResult, StringMatcher

class KMPMatcher(StringMatcher):

    # Knuth-Morris-Pratt: O(n + m).
    # Precompute the LPS (Longest Proper Prefix that is also Suffix) array for the pattern, then scan the text without ever moving the text cursor backwards.

    @staticmethod
    def _build_lps(p: str) -> Tuple[List[int], List[Dict[str, Any]]]:
        m = len(p)
        lps = [0] * m
        length = 0
        i = 1
        lps_steps: List[Dict[str, Any]] = []
        while i < m:
            if p[i] == p[length]:
                length += 1
                lps[i] = length
                lps_steps.append({"i": i, "length": length, "lps": list(lps), "note": f"p[{i}]='{p[i]}' khớp p[{length - 1}], lps[{i}]={length}."})
                i += 1
            else:
                if length != 0:
                    lps_steps.append({"i": i, "length": length, "lps": list(lps), "note": f"p[{i}]='{p[i]}' không khớp, lùi length=lps[{length - 1}]={lps[length - 1]}."})
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    lps_steps.append({"i": i, "length": length, "lps": list(lps), "note": f"Không có prefix-suffix tại i={i}, lps[{i}]=0."})
                    i += 1
        return lps, lps_steps

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

        emit("info", 0, 0, "Bắt đầu thuật toán KMP. Tính mảng LPS cho pattern...")

        if m > n:
            emit("info", 0, 0, "Pattern rỗng hoặc dài hơn Text. Dừng.")
            result.extras["lps"] = []
            return result

        lps, lps_steps = self._build_lps(p)
        result.extras["lps"] = lps
        for s in lps_steps:
            emit("lps", 0, 0, f"[LPS] {s['note']}", lps=s["lps"], lpsI=s["i"])
        emit("info", 0, 0, f"LPS = [{', '.join(str(v) for v in lps)}]. Bắt đầu duyệt text.", lps=lps)

        i = 0
        j = 0
        while i < n:
            result.comparisons += 1
            is_match = t[i] == p[j]
            if is_match:
                emit("match", i - j, j, f"So sánh text[{i}]='{text[i]}' với pattern[{j}]='{pattern[j]}' → khớp.", lps = lps, textCursor = i)
            else:
                emit("mismatch", i - j, j, f"So sánh text[{i}]='{text[i]}' với pattern[{j}]='{pattern[j]}' → không khớp.", lps = lps, textCursor = i)
            if is_match:
                i += 1
                j += 1
                if j == m:
                    result.positions.append(i - j)
                    emit("found", i - j, m - 1, f"Tìm thấy tại vị trí {i - j}. Dùng LPS: j = lps[{j - 1}] = {lps[j - 1]}.", lps = lps)
                    j = lps[j - 1]
            else:
                if j != 0:
                    new_j = lps[j - 1]
                    emit("shift", i - j, new_j, f"Mismatch tại j={j}. Dùng LPS: j = lps[{j - 1}] = {new_j}. Text cursor không lùi.", lps = lps)
                    j = new_j
                else:
                    i += 1

        emit("info", n, 0, f"Hoàn tất KMP. Tìm thấy {len(result.positions)} vị trí với {result.comparisons} phép so sánh.", lps = lps)
        return result
