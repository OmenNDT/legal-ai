import sys
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path để các test có thể
# `from backend.string_matching... import ...` dù chạy pytest từ subfolder.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
