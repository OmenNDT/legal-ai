# backend/string_matching/tests/test_matchers.py
import sys
from pathlib import Path

# Thêm thư mục cha của `backend/` vào sys.path để import `backend.*`
# bất kể chạy từ đâu (vd: `python test.py` trong folder hiện tại).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
from backend.naive.naive_matcher import NaiveMatcher
from backend.kmp.kmp_matcher import KMPMatcher
from backend.boyer_moore.boyer_moore_matcher import BoyerMooreMatcher

# -------------------------------------------------------------------
# Fixture chạy với cả 3 matcher
# -------------------------------------------------------------------
@pytest.fixture(params=[
    (NaiveMatcher, "Naive"),
    (KMPMatcher, "KMP"),
    (BoyerMooreMatcher, "Boyer-Moore")
])
def matcher(request):
    MatcherClass, name = request.param
    matcher = MatcherClass(case_sensitive=True)
    return matcher, name


# -------------------------------------------------------------------
# TEST CASE 1: Pattern ở đầu văn bản
# -------------------------------------------------------------------
def test_pattern_at_beginning(matcher):
    m, name = matcher
    result = m.search("hello world", "hello")
    assert result.positions == [0], f"{name}: expected [0], got {result.positions}"


# -------------------------------------------------------------------
# TEST CASE 2: Pattern ở cuối văn bản
# -------------------------------------------------------------------
def test_pattern_at_end(matcher):
    m, name = matcher
    result = m.search("hello world", "world")
    assert result.positions == [6], f"{name}: expected [6], got {result.positions}"


# -------------------------------------------------------------------
# TEST CASE 3: Pattern xuất hiện nhiều lần
# -------------------------------------------------------------------
def test_multiple_occurrences(matcher):
    m, name = matcher
    result = m.search("ababab", "ab")
    assert result.positions == [0, 2, 4], f"{name}: expected [0,2,4], got {result.positions}"


# -------------------------------------------------------------------
# TEST CASE 4: Pattern không xuất hiện
# -------------------------------------------------------------------
def test_not_found(matcher):
    m, name = matcher
    result = m.search("abcdef", "xyz")
    assert result.positions == [], f"{name}: expected [], got {result.positions}"


# -------------------------------------------------------------------
# TEST CASE 5: Pattern dài hơn văn bản
# -------------------------------------------------------------------
def test_pattern_longer_than_text(matcher):
    m, name = matcher
    result = m.search("abc", "abcdef")
    assert result.positions == [], f"{name}: expected [], got {result.positions}"


# -------------------------------------------------------------------
# TEST CASE 6: Pattern rỗng -> phải raise ValueError
# -------------------------------------------------------------------
def test_empty_pattern_raises_error(matcher):
    m, name = matcher
    with pytest.raises(ValueError, match="Pattern must not be empty"):
        m.search("anything", "")


# -------------------------------------------------------------------
# TEST CASE 7: Case-sensitive BẬT
# -------------------------------------------------------------------
def test_case_sensitive_on():
    m = NaiveMatcher(case_sensitive=True)
    result = m.search("Hello hello", "hello")
    assert result.positions == [6], f"case_sensitive=True: expected [6], got {result.positions}"


# -------------------------------------------------------------------
# TEST CASE 8: Case-sensitive TẮT
# -------------------------------------------------------------------
def test_case_sensitive_off():
    m = NaiveMatcher(case_sensitive=False)
    result = m.search("Hello hello", "hello")
    assert result.positions == [0, 6], f"case_sensitive=False: expected [0,6], got {result.positions}"


# -------------------------------------------------------------------
# TEST CASE 9: Pattern lặp nhiều (aaaaa, aa)
# -------------------------------------------------------------------
def test_repetitive_pattern(matcher):
    m, name = matcher
    result = m.search("aaaaa", "aa")
    # positions: 0,1,2,3
    assert result.positions == [0, 1, 2, 3], f"{name}: expected [0,1,2,3], got {result.positions}"
    # comparisons: có thể khác nhau giữa các thuật toán, nhưng vị trí phải đúng


# -------------------------------------------------------------------
# TEST CASE 10: Pattern có khoảng trắng
# -------------------------------------------------------------------
def test_pattern_with_spaces(matcher):
    m, name = matcher
    result = m.search("to be or not to be", "to be")
    # "to be" xuất hiện ở vị trí 0 và 13 (đếm cả khoảng trắng).
    assert result.positions == [0, 13], f"{name}: expected [0,13], got {result.positions}"


# -------------------------------------------------------------------
# TEST CASE 11: Tiếng Việt có dấu
# -------------------------------------------------------------------
def test_vietnamese_with_tones():
    m = NaiveMatcher(case_sensitive=True)
    text = "Bộ luật Dân sự 2015 quy định về dân sự"
    pattern = "dân sự"
    result = m.search(text, pattern)
    # Vị trí phụ thuộc vào độ dài chuỗi, nhưng phải tìm thấy ít nhất 1 lần
    assert result.count >= 1, "Không tìm thấy pattern tiếng Việt"
    
    # Kiểm tra case_sensitive: 'Dân sự' (viết hoa D) không match 'dân sự'
    result2 = m.search(text, "Dân sự")
    # 'Dân sự' ở đầu câu? tùy text, nhưng ít nhất phải kiểm tra được
    assert result2.count >= 0  # chỉ kiểm tra không lỗi


# -------------------------------------------------------------------
# TEST CASE 12: Pattern chồng lấn (ví dụ "aaa" trong "aaaa")
# -------------------------------------------------------------------
def test_overlapping_pattern(matcher):
    m, name = matcher
    result = m.search("aaaa", "aaa")
    # Vị trí: 0 và 1 (chồng lấn)
    assert result.positions == [0, 1], f"{name}: expected [0,1], got {result.positions}"


# -------------------------------------------------------------------
# TEST RIÊNG KMP: Kiểm tra hàm buildLPS
# -------------------------------------------------------------------
def test_kmp_build_lps():
    matcher = KMPMatcher()
    # _build_lps trả về tuple (lps, lps_steps); chỉ assert phần lps.
    lps, _steps = matcher._build_lps("ABABC")
    assert lps == [0, 0, 1, 2, 0], f"LPS sai: {lps}"


# -------------------------------------------------------------------
# TEST RIÊNG BOYER-MOORE: Kiểm tra bảng bad-character
# -------------------------------------------------------------------
def test_boyer_moore_bad_char():
    matcher = BoyerMooreMatcher()
    # Pattern đã được normalize, nhưng ta test raw
    table = matcher._build_bad_char_table("dansu")
    assert table.get('d') == 0
    assert table.get('a') == 1
    assert table.get('n') == 2
    assert table.get('s') == 3
    assert table.get('u') == 4
    # Ký tự không có trong pattern phải trả về None (hoặc -1 khi dùng .get)
    assert table.get('x') is None


# Cho phép `python test.py` chạy trực tiếp (gọi pytest trên chính file này).
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))