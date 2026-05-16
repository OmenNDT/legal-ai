from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

@dataclass
class MatchResult:
    pattern: str
    text_length: int
    positions: List[int] = field(default_factory=list)
    comparisons: int = 0

    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def found(self) -> bool:
        return self.count > 0

    def __str__(self) -> str:
        return (
            f"MatchResult(pattern={self.pattern!r}, "
            f"count={self.count}, positions={self.positions}, "
            f"comparisons={self.comparisons})"
        )

class StringMatcher(ABC):
    def __init__(self, case_sensitive: bool = True):
        self.case_sensitive = case_sensitive

    def _normalize(self, s: str) -> str:
        return s if self.case_sensitive else s.lower()

    @abstractmethod
    def search(self, text: str, pattern: str) -> MatchResult:
        ...

    def search_first(self, text: str, pattern: str) -> int:
        result = self.search(text, pattern)
        return result.positions[0] if result.found else -1

    def contains(self, text: str, pattern: str) -> bool:
        return self.search_first(text, pattern) != -1

    def highlight(self, text: str, pattern: str, marker: str = "**") -> str:
        result = self.search(text, pattern)
        if not result.found:
            return text
        m = len(pattern)
        out = []
        cursor = 0
        for pos in result.positions:
            out.append(text[cursor:pos])
            out.append(f"{marker}{text[pos:pos + m]}{marker}")
            cursor = pos + m
        out.append(text[cursor:])
        return "".join(out)
