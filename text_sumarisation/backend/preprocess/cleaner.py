import re
import unicodedata

# Lớp làm sạch văn bản hợp đồng trước khi tách câu
class TextCleaner:
    # Một số mẫu noise hay gặp trong CUAD
    PAGE_NUM_RE = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", flags = re.IGNORECASE)
    SOURCE_RE = re.compile(r"Source:\s*[^\n]+", flags = re.IGNORECASE)
    MULTI_DOTS_RE = re.compile(r"\.{3,}")
    MULTI_SPACE_RE = re.compile(r"[ \t]+")
    MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

    def __init__(self, keep_unicode: bool = True):
        self.keep_unicode = keep_unicode

    def clean(self, text: str) -> str:
        # Chuẩn hoá unicode (NFKC)
        text = unicodedata.normalize("NFKC", text)
        # Xoá ký tự không in được
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
        # Xoá đánh số trang và dòng "Source: ..."
        text = self.PAGE_NUM_RE.sub(" ", text)
        text = self.SOURCE_RE.sub(" ", text)
        # Rút gọn dấu chấm liên tiếp
        text = self.MULTI_DOTS_RE.sub(".", text)
        # Gộp khoảng trắng dư
        text = self.MULTI_SPACE_RE.sub(" ", text)
        text = self.MULTI_NEWLINE_RE.sub("\n\n", text)
        # Loại non-ASCII nếu cần (giữ mặc định vì doc tiếng Anh)
        if not self.keep_unicode:
            text = self.NON_ASCII_RE.sub(" ", text)
        return text.strip()
