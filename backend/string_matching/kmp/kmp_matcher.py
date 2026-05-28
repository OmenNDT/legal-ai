from typing import List, Tuple, Dict, Any
from ..base import MatchResult, StringMatcher

class KMPMatcher(StringMatcher):

    # KMP: tiền xử lý LPS để khi mismatch thì nhảy j theo LPS, KHÔNG lùi text cursor.
    # LPS[i] = độ dài chuỗi vừa là prefix vừa là suffix của p[0..i] (không tính chính nó).
    # Độ phức tạp O(n + m) — tối ưu cho worst-case.

    @staticmethod
    def _build_lps(p: str) -> Tuple[List[int], List[Dict[str, Any]]]:
        m = len(p)
        lps = [0] * m # lps[0] = 0 theo định nghĩa
        length = 0 # độ dài prefix-suffix đang giữ
        i = 1 # bắt đầu từ index 1
        lps_steps: List[Dict[str, Any]] = []  # trace để UI animate

        while i < m:
            if p[i] == p[length]:
                # Mở rộng prefix-suffix thêm 1 ký tự.
                length += 1
                lps[i] = length
                lps_steps.append({"i": i, "length": length, "lps": list(lps), "note": f"p[{i}]='{p[i]}' khớp p[{length - 1}], lps[{i}]={length}."})
                i += 1
            else:
                # Mismatch: lùi length về prefix-suffix ngắn hơn (KHÔNG tăng i)
                # cho đến khi length = 0 thì đặt lps[i] = 0.
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

        # Chuẩn hoá (lower-case nếu không phân biệt hoa thường).
        t = self._normalize(text)
        p = self._normalize(pattern)
        n, m = len(t), len(p)

        result = MatchResult(pattern=pattern, text_length=n)

        # Ghi 1 bước vào result.steps (extras kèm lps, textCursor cho UI).
        def emit(kind, i, j, message, **extra):
            if trace:
                step = {
                    "type": kind, # info | lps | match | mismatch | shift | found
                    "i": i, # vị trí align pattern[0] = textCursor - j
                    "j": j, # vị trí trong pattern
                    "comparisons": result.comparisons, # số phép so sánh tích luỹ
                    "positions": list(result.positions), # các vị trí đã tìm thấy
                    "message": message
                }
                step.update(extra)
                result.steps.append(step)

        emit("info", 0, 0, "Bắt đầu thuật toán KMP. Tính mảng LPS cho pattern...")

        if m > n:
            emit("info", 0, 0, "Pattern rỗng hoặc dài hơn Text. Dừng.")
            result.extras["lps"] = []
            return result

        # Bước 1 — xây LPS, O(m).
        lps, lps_steps = self._build_lps(p)
        result.extras["lps"] = lps
        for s in lps_steps:
            emit("lps", 0, 0, f"[LPS] {s['note']}", lps=s["lps"], lpsI=s["i"])
        emit("info", 0, 0, f"LPS = [{', '.join(str(v) for v in lps)}]. Bắt đầu duyệt text.", lps=lps)

        # Bước 2 — duyệt text. i KHÔNG bao giờ lùi (điểm mấu chốt của KMP).
        i = 0 # con trỏ trong text
        j = 0 # con trỏ trong pattern
        while i < n:
            result.comparisons += 1
            is_match = t[i] == p[j]
            if is_match:
                emit("match", i - j, j, f"So sánh text[{i}]='{text[i]}' với pattern[{j}]='{pattern[j]}' → khớp.", lps=lps, textCursor=i)
            else:
                emit("mismatch", i - j, j, f"So sánh text[{i}]='{text[i]}' với pattern[{j}]='{pattern[j]}' → không khớp.", lps=lps, textCursor=i)

            if is_match:
                # Khớp → cả 2 con trỏ tiến.
                i += 1
                j += 1
                if j == m:
                    # Match toàn pattern tại i - j.
                    result.positions.append(i - j)
                    emit("found", i - j, m - 1, f"Tìm thấy tại vị trí {i - j}. Dùng LPS: j = lps[{j - 1}] = {lps[j - 1]}.", lps=lps)
                    # Tiếp tục tìm match overlap qua LPS.
                    j = lps[j - 1]
            else:
                if j != 0:
                    # Mismatch khi j > 0: nhảy j theo LPS, i giữ nguyên.
                    # Đây là lý do KMP không lùi text cursor.
                    new_j = lps[j - 1]
                    emit("shift", i - j, new_j, f"Mismatch tại j={j}. Dùng LPS: j = lps[{j - 1}] = {new_j}. Text cursor không lùi.", lps=lps)
                    j = new_j
                else:
                    # j = 0 mà mismatch → chỉ còn cách dịch i lên 1.
                    i += 1

        emit("info", n, 0, f"Hoàn tất KMP. Tìm thấy {len(result.positions)} vị trí với {result.comparisons} phép so sánh.", lps=lps)
        return result
