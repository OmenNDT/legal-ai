import re
from dataclasses import dataclass
from typing import Iterator

@dataclass
class LawChunk:
    doc_name: str
    phan: str
    chuong: str
    muc: str
    dieu: str
    khoan: str
    diem: str
    content: str

    @property
    def hierarchy_path(self) -> str:
        parts = [p for p in [self.chuong, self.muc, self.dieu, self.khoan, self.diem] if p]
        return " > ".join(parts)

    @property
    def full_text(self) -> str:
        # Chuỗi đầy đủ dùng để nhúng embedding
        parts = [self.doc_name]
        for label in [self.chuong, self.muc, self.dieu, self.khoan, self.diem]:
            if label:
                parts.append(label)
        return ", ".join(parts) + ": " + self.content.strip()

class _Patterns:
    # Regex nhận dạng từng cấp trong phân cấp văn bản luật Việt Nam
    PHAN = re.compile(r"^(Phần\s+(?:thứ\s+)?[IVXLCDM\d]+[:\.\s]*.*)$", re.IGNORECASE)
    CHUONG = re.compile(r"^(Chương\s+[IVXLCDM\d]+[:\.\s]*.*)$", re.IGNORECASE)
    MUC = re.compile(r"^(Mục\s+\d+[:\.\s]*.*)$", re.IGNORECASE)
    DIEU = re.compile(r"^(Điều\s+\d+[\.\s]*.*)$", re.IGNORECASE)
    # Khoản: số thứ tự + dấu chấm, nhưng nội dung phải đủ dài (>= 10 ký tự)
    # để tránh nhận nhầm số liệt kê ngắn bên trong một Khoản
    KHOAN = re.compile(r"^(\d{1,2})\.\s+(.{10,})$")
    DIEM = re.compile(r"^([a-zA-ZđĐ])\)\s+(.+)$")

class HierarchicalParser:
    # State machine bóc tách phân cấp: Phần > Chương > Mục > Điều > Khoản > Điểm
    def __init__(self, doc_name: str) -> None:
        self._doc_name = doc_name
        self._pat = _Patterns()
        self._reset()

    def _reset(self) -> None:
        self._phan = ""
        self._chuong = ""
        self._muc = ""
        self._dieu = ""
        self._khoan = ""
        self._diem = ""
        self._buf: list[str] = []

    def _flush(self) -> LawChunk | None:
        content = " ".join(self._buf).strip()
        self._buf = []
        if not content or not self._dieu:
            return None
        return LawChunk(
            doc_name = self._doc_name,
            phan = self._phan,
            chuong = self._chuong,
            muc = self._muc,
            dieu = self._dieu,
            khoan = self._khoan,
            diem = self._diem,
            content = content
        )

    def parse(self, text: str) -> Iterator[LawChunk]:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            chunk = self._process(line)
            if chunk:
                yield chunk
        last = self._flush()
        if last:
            yield last

    def _process(self, line: str) -> LawChunk | None:
        p = self._pat

        if p.PHAN.match(line):
            chunk = self._flush()
            self._phan = line
            self._chuong = self._muc = self._dieu = self._khoan = self._diem = ""
            return chunk

        if p.CHUONG.match(line):
            chunk = self._flush()
            self._chuong = line
            self._muc = self._dieu = self._khoan = self._diem = ""
            return chunk

        if p.MUC.match(line):
            chunk = self._flush()
            self._muc = line
            self._dieu = self._khoan = self._diem = ""
            return chunk

        if p.DIEU.match(line):
            chunk = self._flush()
            self._dieu  = line
            self._khoan = self._diem = ""
            self._buf.append(line)
            return chunk

        m = p.KHOAN.match(line)
        if m and self._dieu:
            chunk = self._flush()
            self._khoan = f"Khoản {m.group(1)}"
            self._diem = ""
            self._buf.append(m.group(2))
            return chunk

        m = p.DIEM.match(line)
        if m and self._khoan:
            chunk = self._flush()
            self._diem = f"Điểm {m.group(1)}"
            self._buf.append(m.group(2))
            return chunk
        self._buf.append(line)
        return None