from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Đối tượng đại diện cho một hợp đồng đã load lên RAM
@dataclass
class Contract:
    doc_id: str
    file_path: Path
    raw_text: str
    word_count: int = 0
    meta: dict = field(default_factory = dict)

# Lớp đọc các file .txt từ thư mục full_contract_txt
class ContractLoader:
    def __init__(self, txt_dir: Path):
        self.txt_dir = Path(txt_dir)

    # Liệt kê toàn bộ doc_id (tên file bỏ đuôi .txt)
    def list_ids(self) -> List[str]:
        return sorted([p.stem for p in self.txt_dir.glob("*.txt")])

    # Đọc một file theo doc_id
    def load_one(self, doc_id: str) -> Contract:
        path = self.txt_dir / f"{doc_id}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        text = path.read_text(encoding = "utf-8", errors = "ignore")
        return Contract(
            doc_id = doc_id,
            file_path = path,
            raw_text = text,
            word_count = len(text.split())
        )

    # Đọc tất cả 510 file, dùng generator để tiết kiệm RAM
    def iter_all(self):
        for did in self.list_ids():
            yield self.load_one(did)

    # Đọc một danh sách doc_id cụ thể
    def load_many(self, doc_ids: Optional[List[str]] = None) -> List[Contract]:
        ids = doc_ids if doc_ids else self.list_ids()
        return [self.load_one(d) for d in ids]