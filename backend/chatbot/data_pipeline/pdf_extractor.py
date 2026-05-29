import re
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

@dataclass
class RawPage:
    page_number: int
    text: str

class HeaderFooterCleaner:
    _NOISE = [
        re.compile(r"^-+\s*PAGE\s*\d+\s*-+$", re.IGNORECASE),
        re.compile(r"^LuatVietnam.*$", re.IGNORECASE),
        re.compile(r"^\s*\d+\s*$"),
        re.compile(r"^www\.[a-zA-Z0-9._-]+\.[a-z]{2,}$"),
        re.compile(r"^https?://\S+$"),
        re.compile(r"^_{3,}$"),
        re.compile(r"^-{3,}$")
    ]

    def clean(self, text: str) -> str:
        lines = text.splitlines()
        return "\n".join(line for line in lines if not self._is_noise(line))

    def _is_noise(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return any(p.match(stripped) for p in self._NOISE)

class PdfExtractor:
    
    _MIN_TEXT_CHARS = 50

    def __init__(self, cleaner: HeaderFooterCleaner | None = None, ocr_lang: str = "vie", ocr_dpi: int = 200) -> None:
        self._cleaner = cleaner or HeaderFooterCleaner()
        self._ocr_lang = ocr_lang
        self._ocr_dpi = ocr_dpi

    def extract(self, pdf_path: Path) -> Iterator[RawPage]:
        use_ocr = self._needs_ocr(pdf_path)
        if use_ocr:
            print(f"[PdfExtractor] OCR mode: {pdf_path.name}")
            yield from self._extract_ocr(pdf_path)
        else:
            yield from self._extract_native(pdf_path)

    def _needs_ocr(self, pdf_path: Path) -> bool:
        with pdfplumber.open(pdf_path) as pdf:
            sample = pdf.pages[:3]
            total_chars = sum(len(p.extract_text() or "") for p in sample)
        return total_chars < self._MIN_TEXT_CHARS * len(sample)

    def _extract_native(self, pdf_path: Path) -> Iterator[RawPage]:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ""
                yield RawPage(
                    page_number = page.page_number,
                    text = self._cleaner.clean(raw)
                )

    def _extract_ocr(self, pdf_path: Path) -> Iterator[RawPage]:
        images = convert_from_path(pdf_path, dpi = self._ocr_dpi)
        for i, img in enumerate(images):
            raw = pytesseract.image_to_string(img, lang = self._ocr_lang)
            yield RawPage(
                page_number = i + 1,
                text = self._cleaner.clean(raw)
            )

    def extract_full_text(self, pdf_path: Path) -> str:
        return "\n".join(p.text for p in self.extract(pdf_path))