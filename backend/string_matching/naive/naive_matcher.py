from ..base import MatchResult, StringMatcher

class NaiveMatcher(StringMatcher):

    # Naive: trượt pattern từng ô, so sánh trái→phải. Mismatch ở đâu cũng trượt 1.
    # Độ phức tạp O(n*m). Không học gì từ lần so sánh trước.

    def search(self, text: str, pattern: str, trace: bool = False) -> MatchResult:
        # Pattern rỗng → vô nghĩa.
        if pattern == "":
            raise ValueError(">>> Pattern must not be empty!")

        # Chuẩn hoá để so sánh (lower-case nếu case_sensitive=False).
        # Giữ text/pattern gốc để in message cho UI.
        t = self._normalize(text)
        p = self._normalize(pattern)
        n, m = len(t), len(p)  # n = len text, m = len pattern

        result = MatchResult(pattern=pattern, text_length=n)

        # Ghi 1 "bước" vào result.steps để UI animate (chỉ khi trace=True).
        def emit(kind, i, j, message):
            if trace:
                result.steps.append({
                    "type": kind,                        # info | match | mismatch | found
                    "i": i,                              # vị trí trượt trong text
                    "j": j,                              # vị trí trong pattern
                    "comparisons": result.comparisons,   # số phép so sánh tích luỹ
                    "positions": list(result.positions), # các vị trí đã tìm thấy
                    "message": message
                })

        emit("info", 0, 0, f"Bắt đầu thuật toán Naive (Brute-force). Pattern dài {m}, Text dài {n}.")

        # Pattern dài hơn text → không thể match.
        if m > n:
            emit("info", 0, 0, "Pattern rỗng hoặc dài hơn Text. Dừng.")
            return result

        # Trượt pattern qua text: i là vị trí align pattern[0].
        # i chạy đến n-m vì cần đủ m ký tự còn lại.
        for i in range(n - m + 1):
            emit("info", i, 0, f"Trượt pattern đến vị trí i={i} trong text.")
            j = 0
            # So sánh từng ký tự pattern với text bắt đầu từ i.
            while j < m:
                result.comparisons += 1
                is_match = t[i + j] == p[j]
                if is_match:
                    emit("match", i, j, f"So sánh text[{i + j}]='{text[i + j]}' với pattern[{j}]='{pattern[j]}' → khớp.")
                else:
                    emit("mismatch", i, j, f"So sánh text[{i + j}]='{text[i + j]}' với pattern[{j}]='{pattern[j]}' → không khớp. Trượt sang phải 1.")
                # Mismatch → bỏ vị trí i, sang i+1.
                if not is_match:
                    break
                j += 1
            # j chạm m → khớp toàn pattern tại i.
            if j == m:
                result.positions.append(i)
                emit("found", i, m - 1, f"Tìm thấy pattern tại vị trí {i}.")

        emit("info", n, 0, f"Hoàn tất. Tìm thấy {len(result.positions)} vị trí với {result.comparisons} phép so sánh.")
        return result
