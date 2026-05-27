import ast
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Trích xuất các clause từ master_clauses.csv để dùng làm bản tóm tắt tham chiếu
class ReferenceBuilder:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self._df: Optional[pd.DataFrame] = None
        self._index = None

    # Đọc CSV và trả về DataFrame đã sẵn sàng (lazy load)
    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self.csv_path)
            df.columns = [c.strip() for c in df.columns]
            # Tên file trong CSV là *.pdf, ta dùng phần stem để khớp với .txt
            if "Filename" in df.columns:
                df["doc_id"] = df["Filename"].apply(lambda x: Path(str(x)).stem)
            else:
                df["doc_id"] = df.iloc[:, 0].apply(lambda x: Path(str(x)).stem)
            self._df = df
        return self._df

    # Tìm các cột chứa clause (không phải cột Answer)
    def _clause_columns(self) -> List[str]:
        df = self._ensure_loaded()
        cols = []
        for c in df.columns:
            if c in ("Filename", "doc_id"):
                continue
            if c.lower().endswith("-answer"):
                continue
            cols.append(c)
        return cols

    # Một số ô có dạng list Python dạng chuỗi - parse an toàn
    @staticmethod
    def _parse_cell(val) -> List[str]:
        if val is None:
            return []
        if isinstance(val, float):
            return []
        s = str(val).strip()
        if not s or s.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
            return [str(parsed).strip()]
        except Exception:
            # Tách thô theo dấu chấm phẩy hoặc xuống dòng
            return [p.strip() for p in re.split(r"[\n;]+", s) if p.strip()]

    # Lấy reference text của một doc bằng cách gộp tất cả clause
    def get_reference(self, doc_id: str) -> str:
        df = self._ensure_loaded()
        row = df[df["doc_id"] == doc_id]
        if row.empty:
            return ""
        row0 = row.iloc[0]
        parts = []
        for col in self._clause_columns():
            parts.extend(self._parse_cell(row0.get(col)))
        # Khử trùng lặp giữ thứ tự
        seen, uniq = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return " ".join(uniq)

    # Bản đồ doc_id -> reference cho toàn bộ
    def all_references(self) -> Dict[str, str]:
        df = self._ensure_loaded()
        return {did: self.get_reference(did) for did in df["doc_id"].tolist()}
