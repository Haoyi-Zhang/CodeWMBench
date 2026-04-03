from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BLOCK_COMMENT_RE = re.compile(r"(?s)/\*.*?\*/")


def stable_hash(text: str, *, secret: str = "", digest_size: int = 12) -> str:
    hasher = hashlib.blake2b(digest_size=digest_size, key=secret.encode("utf-8"))
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def language_comment_prefix(language: str) -> str:
    language = language.lower()
    if language in {"c", "c++", "cpp", "java", "js", "javascript", "ts", "typescript", "go", "rust"}:
        return "//"
    return "#"


def strip_comments(text: str) -> str:
    text = _BLOCK_COMMENT_RE.sub("", text)
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line
        for marker in ("#", "//"):
            index = stripped.find(marker)
            if index != -1:
                stripped = stripped[:index]
        stripped = stripped.rstrip()
        if stripped.strip():
            lines.append(stripped)
    return "\n".join(lines)


def normalize_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(line for line in lines if line.strip())


def line_count(text: str) -> int:
    return len(text.splitlines())


def edit_distance_ratio(left: str, right: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(a=left, b=right).ratio()


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union)


def scrub_paths(text: str) -> str:
    text = re.sub(r"[A-Za-z]:\\[^\\\s'\"]+(?:\\[^\\\s'\"]+)+", "<path>", text)
    text = re.sub(r"/(?:[^/\s]+/)+[^/\s]+", "<path>", text)
    return text


@dataclass(frozen=True, slots=True)
class StableRandom:
    seed: int

    def choice(self, items: list[str]) -> str:
        if not items:
            raise ValueError("choice requires a non-empty list")
        index = self.seed % len(items)
        return items[index]

    def shuffle(self, items: list[Any]) -> list[Any]:
        result = list(items)
        n = len(result)
        for idx in range(n):
            swap = (self.seed + idx * 17) % n if n else 0
            result[idx], result[swap] = result[swap], result[idx]
        return result


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
