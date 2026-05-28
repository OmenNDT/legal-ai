"""
Section-aware Hierarchical Chunker for technical standards (ASME, API, ASTM).

Splits documents at section boundaries instead of fixed character counts.
Preserves hierarchy metadata (chapter, section, parent) and extracts cross-references.
"""
import re
from dataclasses import dataclass, field

# Section header patterns (ordered by priority)
# Matches: "## 5.3.3.2 Title", "**5.3.3.2** Title", "5.3.3.2 Title", "Article 5", "Chapter 3"
SECTION_PATTERNS = [
    # Markdown heading with optional section number
    re.compile(r'^(#{1,6})\s+([\d]+(?:\.[\d]+)*\.?\s+.+)', re.MULTILINE),
    # Markdown heading without number
    re.compile(r'^(#{1,6})\s+(.+)', re.MULTILINE),
    # Bold numbered section: **1. Title** or **5.3.3.2 Title**
    re.compile(r'^\*\*(\d+(?:\.\d+)*\.?)\s+(.+?)\*\*', re.MULTILINE),
    # Plain numbered section at line start: 5.3.3.2 Title
    re.compile(r'^(\d+(?:\.\d+)+\.?)\s+([A-Z].+)', re.MULTILINE),
    # Article/Chapter markers
    re.compile(r'^(Article|Chapter|Part|Annex|Appendix)\s+(\w+)', re.MULTILINE | re.IGNORECASE),
]

# Cross-reference patterns
XREF_PATTERNS = [
    re.compile(r'(?:see|refer\s+to|per|theo|xem)\s+(?:Section\s+|Mục\s+)?(\d+(?:\.\d+)+)', re.IGNORECASE),
    re.compile(r'(?:Article|Chapter|Part)\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'(?:Table|Figure|Exhibit)\s+(\w+(?:\.\d+)*)', re.IGNORECASE),
]

MAX_CHUNK_SIZE = 1200  # Max chars per chunk (larger than naive since sections are meaningful)
MIN_CHUNK_SIZE = 80    # Skip chunks shorter than this


@dataclass
class SectionChunk:
    """A chunk with hierarchical metadata."""
    text: str
    section_id: str = ""          # e.g. "5.3.3.2"
    section_title: str = ""       # e.g. "Welding Requirements"
    chapter: str = ""             # e.g. "5" or "Chapter 5"
    parent_sections: list = field(default_factory=list)  # e.g. ["5", "5.3", "5.3.3"]
    cross_references: list = field(default_factory=list) # e.g. ["3.2.4", "Table 1"]
    heading_level: int = 0        # 1-6 for markdown, 0 for inferred
    chunk_index: int = 0


def _extract_cross_references(text: str) -> list[str]:
    """Extract cross-reference targets from text."""
    refs = set()
    for pattern in XREF_PATTERNS:
        for match in pattern.finditer(text):
            refs.add(match.group(1))
    return sorted(refs)


def _get_parent_sections(section_id: str) -> list[str]:
    """Get parent section IDs. E.g. '5.3.3.2' → ['5', '5.3', '5.3.3']"""
    parts = section_id.rstrip('.').split('.')
    parents = []
    for i in range(1, len(parts)):
        parents.append('.'.join(parts[:i]))
    return parents


def _extract_section_number(text: str) -> str:
    """Try to extract a section number from heading text. E.g. '5.3.3.2 Scope' → '5.3.3.2'"""
    match = re.match(r'(\d+(?:\.\d+)*\.?)\s', text.strip())
    return match.group(1).rstrip('.') if match else ""


def _split_long_section(text: str, max_size: int) -> list[str]:
    """Split a long section by paragraphs, then by sentences if still too long."""
    if len(text) <= max_size:
        return [text]

    # Try splitting by double-newline (paragraphs)
    paragraphs = re.split(r'\n\n+', text)
    if len(paragraphs) > 1:
        return _merge_small_parts(paragraphs, max_size)

    # Fallback: split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return _merge_small_parts(sentences, max_size)


def _merge_small_parts(parts: list[str], max_size: int) -> list[str]:
    """Merge small parts into chunks up to max_size."""
    chunks = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) + 2 > max_size:
            chunks.append(current.strip())
            current = part
        else:
            current = current + "\n\n" + part if current else part
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_by_sections(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> list[SectionChunk]:
    """
    Split text into chunks at section boundaries.

    Strategy:
    1. Find all section headers in text
    2. Split text at each header position
    3. Each section becomes a chunk (or multiple if too long)
    4. Preserve metadata: section_id, chapter, cross-refs
    5. Fallback to paragraph-based splitting if no sections detected
    """
    if not text or not text.strip():
        return []

    # Find all section boundaries
    boundaries = []
    for pattern in SECTION_PATTERNS:
        for match in pattern.finditer(text):
            start = match.start()
            full_match = match.group(0)
            groups = match.groups()

            # Determine heading level and section info
            level = 0
            section_id = ""
            title = full_match.strip()

            if groups[0].startswith('#'):
                level = len(groups[0])
                title = groups[1].strip()
                section_id = _extract_section_number(title)
            elif groups[0].lower() in ('article', 'chapter', 'part', 'annex', 'appendix'):
                level = 1
                section_id = groups[1]
                title = full_match.strip()
            elif re.match(r'\d', groups[0]):
                section_id = groups[0].rstrip('.')
                level = min(len(section_id.split('.')), 6)
                title = groups[1].strip() if len(groups) > 1 else ""

            # Avoid duplicate boundaries at same position
            if not any(abs(b[0] - start) < 5 for b in boundaries):
                boundaries.append((start, level, section_id, title))

    # Sort by position
    boundaries.sort(key=lambda b: b[0])

    # If no section boundaries found, fall back to paragraph-based chunking
    if not boundaries:
        return _fallback_paragraph_chunks(text, max_chunk_size)

    # Split text at boundaries
    chunks = []
    chunk_idx = 0

    for i, (start, level, section_id, title) in enumerate(boundaries):
        # End of this section = start of next section (or end of text)
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) < MIN_CHUNK_SIZE:
            continue

        # Determine chapter from section_id
        chapter = section_id.split('.')[0] if section_id and '.' in section_id else section_id

        # Extract cross-references
        xrefs = _extract_cross_references(section_text)

        # Split if too long
        sub_texts = _split_long_section(section_text, max_chunk_size)

        for sub_text in sub_texts:
            if len(sub_text) < MIN_CHUNK_SIZE:
                continue
            chunks.append(SectionChunk(
                text=sub_text,
                section_id=section_id,
                section_title=title,
                chapter=chapter,
                parent_sections=_get_parent_sections(section_id) if section_id else [],
                cross_references=xrefs,
                heading_level=level,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

    # Handle text before first boundary (preamble)
    if boundaries and boundaries[0][0] > MIN_CHUNK_SIZE:
        preamble = text[:boundaries[0][0]].strip()
        if len(preamble) >= MIN_CHUNK_SIZE:
            xrefs = _extract_cross_references(preamble)
            preamble_chunk = SectionChunk(
                text=preamble,
                section_id="preamble",
                section_title="Preamble",
                chapter="0",
                cross_references=xrefs,
                chunk_index=-1,
            )
            chunks.insert(0, preamble_chunk)
            # Re-index
            for i, c in enumerate(chunks):
                c.chunk_index = i

    return chunks if chunks else _fallback_paragraph_chunks(text, max_chunk_size)


def _fallback_paragraph_chunks(text: str, max_chunk_size: int) -> list[SectionChunk]:
    """Fallback: split by paragraphs when no section headers found."""
    paragraphs = re.split(r'\n\n+', text)
    merged = _merge_small_parts(paragraphs, max_chunk_size)

    chunks = []
    for i, part in enumerate(merged):
        if len(part) < MIN_CHUNK_SIZE:
            continue
        xrefs = _extract_cross_references(part)
        chunks.append(SectionChunk(
            text=part,
            section_id=f"para_{i}",
            section_title="",
            cross_references=xrefs,
            chunk_index=i,
        ))
    return chunks
