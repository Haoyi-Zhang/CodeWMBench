from __future__ import annotations

from pathlib import Path


LANGUAGE_ALIASES: dict[str, str] = {
    "c++": "cpp",
    "cplusplus": "cpp",
    "go/golang": "go",
    "golang": "go",
    "javascript": "javascript",
    "js": "javascript",
    "python3": "python",
}


LANGUAGE_FAMILIES: dict[str, str] = {
    "python": "python",
    "javascript": "ecmascript",
    "typescript": "ecmascript",
    "java": "jvm",
    "cpp": "systems",
    "c": "systems",
    "go": "systems",
    "rust": "systems",
}


RUNNER_IMAGES: dict[str, str] = {
    "python": "python:3.11",
    "cpp": "gcc:13",
    "java": "openjdk:21",
    "javascript": "node:20",
    "go": "golang:1.22",
    "rust": "rust:1.78",
}


LANGUAGE_VERSIONS: dict[str, str] = {
    "python": "3.11",
    "cpp": "c++17",
    "java": "21",
    "javascript": "20",
    "go": "1.22",
    "rust": "1.78",
}


VALIDATION_MODES: dict[str, str] = {
    "python": "python_exec",
    "cpp": "docker_remote",
    "java": "docker_remote",
    "javascript": "docker_remote",
    "go": "docker_remote",
    "rust": "docker_remote",
}


def normalize_language_name(language: str) -> str:
    normalized = str(language or "").strip().lower()
    if not normalized:
        return "unknown"
    return LANGUAGE_ALIASES.get(normalized, normalized)


def language_family(language: str) -> str:
    normalized = normalize_language_name(language)
    return LANGUAGE_FAMILIES.get(normalized, normalized or "unknown")


def validation_mode(language: str) -> str:
    normalized = normalize_language_name(language)
    return VALIDATION_MODES.get(normalized, "unavailable")


def default_evaluation_backend(language: str) -> str:
    mode = validation_mode(language)
    if mode == "python_exec":
        return "python_exec"
    if mode == "docker_remote":
        return "docker_remote"
    return "unavailable"


def runner_image(language: str) -> str:
    return RUNNER_IMAGES.get(normalize_language_name(language), "")


def language_version(language: str) -> str:
    return LANGUAGE_VERSIONS.get(normalize_language_name(language), "")


def supports_execution(language: str, tests: tuple[str, ...] | list[str], *, backend: str | None = None) -> bool:
    if not tests:
        return False
    selected = str(backend or default_evaluation_backend(language)).strip().lower()
    return selected in {"python_exec", "docker_remote", "local_cpp", "mock_multilingual"}


def default_problem_filename(language: str) -> str:
    normalized = normalize_language_name(language)
    suffix = {
        "python": "solution.py",
        "cpp": "solution.cpp",
        "java": "Solution.java",
        "javascript": "solution.js",
        "go": "main.go",
        "rust": "main.rs",
    }.get(normalized, "solution.txt")
    return suffix


def source_relative_to(root: Path, path: Path) -> str:
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.resolve().relative_to(root)
    except Exception:
        return candidate.name
    if ".coordination" in relative.parts:
        return relative.name
    if len(relative.parts) >= 3 and relative.parts[0] == "data" and relative.parts[1] == "public" and relative.parts[2] == "_cache":
        return relative.name
    return str(relative)
