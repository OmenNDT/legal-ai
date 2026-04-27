"""Parse Luật Kế toán 2025 into structured JSON hierarchy.

Output: data/processed/luat_ke_toan_2025_structured.json
Structure: Chương → Điều → Khoản → Điểm
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Diem:
    id: str
    content: str


@dataclass
class Khoan:
    id: str
    content: str
    diems: list


@dataclass
class Dieu:
    id: str
    title: str
    content: str
    khoans: list


@dataclass
class Chuong:
    id: str
    title: str
    dieus: list


def parse_law(text: str) -> list:
    lines = text.split("\n")
    chapters = []
    current_chuong = None
    current_dieu = None
    current_khoan = None
    current_diem = None
    buffer = []

    def save_current_diem():
        nonlocal current_diem, current_khoan
        if current_diem and current_khoan:
            current_khoan.diems.append(current_diem)
            current_diem = None

    def save_current_khoan():
        nonlocal current_khoan, current_dieu
        save_current_diem()
        if current_khoan and current_dieu:
            current_dieu.khoans.append(current_khoan)
            current_khoan = None

    def save_current_dieu():
        nonlocal current_dieu, current_chuong
        save_current_khoan()
        if current_dieu and current_chuong:
            current_chuong.dieus.append(current_dieu)
            current_dieu = None

    def save_current_chuong():
        nonlocal current_chuong
        save_current_dieu()
        if current_chuong:
            chapters.append(current_chuong)
            current_chuong = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Detect Chương
        chuong_match = re.match(r"^Chương\s+([IVX0-9]+)$", line)
        if chuong_match:
            save_current_chuong()
            title_lines = []
            j = i + 1
            while j < len(lines) and lines[j].strip() and not re.match(r"^(Chương\s|Điều\s|Mục\s|Phụ lục)", lines[j].strip()):
                title_lines.append(lines[j].strip())
                j += 1
            chuong_title = " ".join(title_lines) if title_lines else ""
            current_chuong = Chuong(
                id=chuong_match.group(1),
                title=chuong_title,
                dieus=[]
            )
            i = j
            continue

        # Detect Mục (skip, treat as part of chapter)
        muc_match = re.match(r"^Mục\s+([0-9]+)$", line)
        if muc_match:
            i += 1
            continue

        # Detect Điều
        dieu_match = re.match(r"^Điều\s+([0-9]+)\.\s*(.+)$", line)
        if dieu_match:
            save_current_dieu()
            dieu_id = dieu_match.group(1)
            dieu_title = dieu_match.group(2).strip()
            current_dieu = Dieu(
                id=dieu_id,
                title=dieu_title,
                content="",
                khoans=[]
            )
            i += 1
            continue

        # Detect Khoản
        khoan_match = re.match(r"^([0-9]+)\.\s+(.+)$", line)
        if khoan_match and current_dieu is not None:
            save_current_khoan()
            khoan_id = khoan_match.group(1)
            khoan_text = khoan_match.group(2)
            current_khoan = Khoan(
                id=khoan_id,
                content=khoan_text,
                diems=[]
            )
            i += 1
            continue

        # Detect Điểm (a), b), đ), c), d)...)
        diem_match = re.match(r"^([a-zđ])\)\s+(.+)$", line, re.IGNORECASE)
        if diem_match and current_khoan is not None:
            save_current_diem()
            diem_id = diem_match.group(1).lower()
            diem_text = diem_match.group(2)
            current_diem = Diem(
                id=diem_id,
                content=diem_text
            )
            i += 1
            continue

        # Regular content
        if current_diem is not None:
            current_diem.content += " " + line
        elif current_khoan is not None:
            current_khoan.content += " " + line
        elif current_dieu is not None:
            if current_dieu.content:
                current_dieu.content += " " + line
            else:
                current_dieu.content = line

        i += 1

    save_current_chuong()
    return chapters


def chapters_to_dict(chapters: list) -> list:
    result = []
    for ch in chapters:
        ch_dict = {
            "id": ch.id,
            "title": ch.title,
            "dieus": []
        }
        for d in ch.dieus:
            dieu_dict = {
                "id": d.id,
                "title": d.title,
                "content": d.content,
                "khoans": []
            }
            for k in d.khoans:
                khoan_dict = {
                    "id": k.id,
                    "content": k.content,
                    "diems": [{"id": di.id, "content": di.content} for di in k.diems]
                }
                dieu_dict["khoans"].append(khoan_dict)
            ch_dict["dieus"].append(dieu_dict)
        result.append(ch_dict)
    return result


def main():
    raw_path = Path("data/raw/luat_ke_toan_2025.txt")
    out_path = Path("data/processed/luat_ke_toan_2025_structured.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {raw_path}...")
    text = raw_path.read_text(encoding="utf-8")

    print("Parsing law structure...")
    chapters = parse_law(text)

    data = chapters_to_dict(chapters)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_dieu = sum(len(ch["dieus"]) for ch in data)
    total_khoan = sum(len(d["khoans"]) for ch in data for d in ch["dieus"])
    total_diem = sum(len(di) for ch in data for d in ch["dieus"] for k in d["khoans"] for di in k["diems"])

    print(f"Parsed {len(data)} chapters, {total_dieu} articles, {total_khoan} clauses, {total_diem} points")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
