from typing import Dict, List
from ..base import MatchResult, StringMatcher

class BoyerMooreMatcher(StringMatcher):

    # Boyer-Moore (bad-character + good-suffix).
    # So sánh pattern PHẢI → TRÁI. Khi mismatch, dịch theo MAX của 2 luật:
    #   1. bad-character: dùng vị trí cuối của ký tự text bị mismatch trong pattern.
    #   2. good-suffix : dùng đoạn suffix đã khớp để tìm lần xuất hiện khác trong pattern.
    # Hai luật bù trừ cho nhau → bước trượt lớn, ít so sánh.

    @staticmethod
    def _build_bad_char_table(p: str) -> Dict[str, int]:
        # bad_char[c] = chỉ số LỚN NHẤT mà c xuất hiện trong pattern.
        # Ký tự không có → trả -1 khi tra (dùng dict.get(c, -1)).
        table: Dict[str, int] = {}
        for i, ch in enumerate(p):
            table[ch] = i # ghi đè → giữ vị trí cuối cùng
        return table

    @staticmethod
    def _build_good_suffix_table(p: str) -> List[int]:
        # good_suffix[j] = số bước cần dịch khi mismatch ở pattern[j]
        # (lúc đó suffix p[j+1..m-1] đã khớp với text).
        # Cài đặt theo Crochemore-Lecroq, gồm 2 giai đoạn (Case 1 & Case 2).

        m = len(p)

        # Giai đoạn 1 — mảng suffix:
        # suffix[i] = độ dài đoạn dài nhất kết thúc tại i mà cũng là suffix của pattern.
        suffix = [0] * m
        suffix[m - 1] = m
        g = m - 1 # ranh giới trái của "border" đang xét
        f = m - 1 # mốc so sánh; gán mặc định để Pylance không cảnh báo
        for i in range(m - 2, -1, -1):
            if i > g and suffix[i + m - 1 - f] < i - g:
                # Tận dụng kết quả đã tính (giống Z-array).
                suffix[i] = suffix[i + m - 1 - f]
            else:
                if i < g:
                    g = i
                f = i
                # Mở rộng border bằng cách so sánh ngược về 0.
                while g >= 0 and p[g] == p[g + m - 1 - f]:
                    g -= 1
                suffix[i] = f - g

        # Giai đoạn 2 — biến suffix[] thành good_suffix[]:
        # Mặc định dịch toàn pattern (m).
        good_suffix = [m] * m

        # Case 2: suffix đã khớp KHÔNG xuất hiện lại trong pattern,
        # nhưng prefix của pattern trùng với 1 đoạn cuối của suffix đó.
        j = 0
        for i in range(m - 1, -1, -1):
            if suffix[i] == i + 1:
                # p[0..i] cũng là suffix của pattern → dùng làm điểm align mới.
                while j < m - 1 - i:
                    if good_suffix[j] == m:
                        good_suffix[j] = m - 1 - i
                    j += 1

        # Case 1: suffix đã khớp XUẤT HIỆN ở chỗ khác trong pattern → align về đó.
        for i in range(m - 1):
            good_suffix[m - 1 - suffix[i]] = m - 1 - i

        return good_suffix

    def search(self, text: str, pattern: str, trace: bool = False) -> MatchResult:
        if pattern == "":
            raise ValueError(">>> Pattern must not be empty!")

        # Chuẩn hoá; giữ bản gốc cho message UI.
        t = self._normalize(text)
        p = self._normalize(pattern)
        n, m = len(t), len(p)

        result = MatchResult(pattern=pattern, text_length=n)

        # Ghi 1 bước vào result.steps (kèm badChar, goodSuffix, shiftBy... cho UI).
        def emit(kind, i, j, message, **extra):
            if trace:
                step = {
                    "type": kind, # info | match | mismatch | shift | found
                    "i": i, # = s, vị trí align pattern[0]
                    "j": j, # vị trí trong pattern (từ m-1 giảm dần)
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

        # Tiền xử lý: bảng bad-char + bảng good-suffix.
        bad_char = self._build_bad_char_table(p)
        good_suffix = self._build_good_suffix_table(p)
        result.extras["badChar"] = bad_char
        result.extras["goodSuffix"] = good_suffix
        emit("info", 0, 0, f"Bảng bad-character: {bad_char}. Bảng good-suffix: {good_suffix}. So sánh pattern từ PHẢI sang TRÁI.", badChar=bad_char, goodSuffix=good_suffix)

        # s = shift của pattern so với text (pattern[0] align với text[s]).
        s = 0
        while s <= n - m:
            # So sánh từ KÝ TỰ CUỐI pattern, giảm j về 0.
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
                    break # mismatch → tính shift
                j -= 1 # khớp → sang trái

            if j < 0:
                # Match toàn pattern: dịch theo good_suffix[0].
                result.positions.append(s)
                next_shift = good_suffix[0] if good_suffix[0] < m else m
                emit("found", s, 0, f"Tìm thấy tại vị trí {s}. Trượt thêm {next_shift} theo good-suffix rule.", badChar=bad_char, goodSuffix=good_suffix, shiftBy=next_shift)
                s += next_shift
            else:
                # Mismatch tại p[j] vs t[s+j].
                # Bad-char shift = j - bad_char[mismatch_char], chặn min 1 để luôn tiến.
                bc = bad_char.get(t[s + j], -1)
                bc_shift = max(1, j - bc)

                if j == m - 1:
                    # Mismatch ngay ký tự cuối → chưa có suffix khớp → chỉ dùng bad-char.
                    shift_by = bc_shift
                    emit("shift", s, j, f"Không khớp ngay tại j={j}. Bad-char '{text[s + j]}' tại pattern index {bc}. Trượt sang phải {shift_by} bước (chỉ bad-character).", badChar=bad_char, goodSuffix=good_suffix, bcShift=bc_shift, gsShift=0, shiftBy=shift_by)
                else:
                    # Đã có suffix p[j+1..m-1] khớp → so sánh 2 shift, lấy MAX.
                    gs_shift = good_suffix[j]
                    shift_by = max(bc_shift, gs_shift)
                    rule_used = "bad-character" if bc_shift >= gs_shift else "good-suffix"
                    emit("shift", s, j, f"Khớp 1 phần (suffix từ j+1 đã khớp). Bad-char shift={bc_shift}, Good-suffix shift={gs_shift}. Trượt sang phải {shift_by} bước (theo {rule_used}).", badChar=bad_char, goodSuffix=good_suffix, bcShift=bc_shift, gsShift=gs_shift, shiftBy=shift_by)

                s += shift_by

        emit("info", n, 0, f"Hoàn tất Boyer-Moore. Tìm thấy {len(result.positions)} vị trí với {result.comparisons} phép so sánh.", badChar=bad_char, goodSuffix=good_suffix)
        return result
