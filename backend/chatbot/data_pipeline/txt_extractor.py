import re
from pathlib import Path
from dataclasses import dataclass

@dataclass
class RawPage:
    page_number: int
    text: str

class TxtCleaner:
    # Ký tự đặc biệt bị lỗi encoding từ PDF-to-txt conversion
    _NOT_SIGN = re.compile(r"¬") # U+00AC chèn vào giữa chữ (Luat_Bao_Hiem_Xa_Hoi)
    _SOFT_HYPHEN = re.compile(r"\xad") # U+00AD soft-hyphen (Bo_Luat_To_Tung_Hinh_Su)

    # Smart quotes → dấu nháy thường
    _SMART_QUOTE_OPEN = re.compile(r'[""„‟]')
    _SMART_QUOTE_CLOSE = re.compile(r"['']")

    # En-dash dùng như dấu gạch nối trong tên cơ quan → chuẩn hóa thành " - "
    _EN_DASH = re.compile(r"\s*–\s*")

    # Dòng separator thuần ký tự gạch
    _SEPARATOR = re.compile(r"^[-=_]{3,}\s*$")

    # Header văn bản pháp luật (2 cột dạng tab: "Luật số: xxx\tCỘNG HÒA...")
    _HEADER_LAW_NUMBER = re.compile(
        r"^(?:Luật số|Số)\s*:\s*[\w/]+\t.*$", re.IGNORECASE
    )
    # Dòng chỉ chứa "QUỐC HỘI" hoặc tiêu đề đầu trang
    _HEADER_INSTITUTION = re.compile(
        r"^(?:QUỐC HỘI|ỦY BAN THƯỜNG VỤ QUỐC HỘI|CHỦ TỊCH QUỐC HỘI)\s*$"
    )
    # Dòng chữ ký cuối văn bản
    _SIGNATURE_LINE = re.compile(
        r"^\s*(?:\t+)?\s*(?:CHỦ TỊCH QUỐC HỘI|TM\.\s*|Nơi nhận\s*:)\s*$"
    )
    # Marker kết thúc nội dung pháp lý — mọi thứ sau đây là boilerplate chữ ký
    _END_OF_LAW = re.compile(
        r"^(?:Luật|Bộ luật|Pháp lệnh) này đã được Quốc hội.*thông qua"
    )
    # Dòng tab đơn hoặc khoảng trắng đơn (dòng trống giả)
    _BLANK_ONLY = re.compile(r"^\s+$")

    def clean_text(self, text: str) -> str:
        # 1. Xoá soft-hyphen và ký tự NOT SIGN lỗi encoding
        text = self._SOFT_HYPHEN.sub("", text)
        text = self._NOT_SIGN.sub("", text)

        # 2. Chuẩn hóa smart quotes
        text = self._SMART_QUOTE_OPEN.sub('"', text)
        text = self._SMART_QUOTE_CLOSE.sub("'", text)

        # 3. Chuẩn hóa en-dash
        text = self._EN_DASH.sub(" - ", text)

        # 4. Xử lý từng dòng
        lines = text.splitlines()
        cleaned = []
        end_reached = False
        for line in lines:
            # Thay tab bằng khoảng trắng đơn
            line = line.replace("\t", " ")
            # Rút gọn nhiều space liên tiếp
            line = re.sub(r" {2,}", " ", line)
            # Bỏ trailing space
            line = line.rstrip()

            stripped = line.strip()

            # Giữ lại dòng "thông qua..." nhưng đánh dấu kết thúc nội dung
            if not end_reached and self._END_OF_LAW.match(stripped):
                cleaned.append(line)
                end_reached = True
                continue

            # Bỏ toàn bộ boilerplate sau câu "thông qua"
            if end_reached:
                continue

            # Bỏ dòng noise
            if self._SEPARATOR.match(stripped):
                continue
            if self._HEADER_LAW_NUMBER.match(stripped):
                continue
            if self._HEADER_INSTITUTION.match(stripped):
                continue
            if self._SIGNATURE_LINE.match(line):
                continue
            if self._BLANK_ONLY.match(line) and line != "":
                line = ""

            cleaned.append(line)

        # 5. Thu gọn ≥3 dòng trống liên tiếp thành tối đa 2
        result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
        return result.strip()


class TxtExtractor:
    
    # Đọc file .txt và trả về một RawPage duy nhất (cả file là một trang).

    def __init__(self, cleaner: TxtCleaner | None = None) -> None:
        self._cleaner = cleaner or TxtCleaner()

    def extract_full_text(self, txt_path: Path) -> str:
        raw = txt_path.read_text(encoding = "utf-8")
        return self._cleaner.clean_text(raw)

    def extract(self, txt_path: Path) -> list[RawPage]:
        return [RawPage(page_number = 1, text = self.extract_full_text(txt_path))]
