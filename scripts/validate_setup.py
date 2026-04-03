from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _hf_readiness import HFModelRequirement, validate_local_hf_cache
from codewmbench.attacks.registry import available_attacks
from codewmbench.baselines.stone_family.common import (
    stone_family_checkout_metadata,
    validate_checkout as validate_runtime_checkout,
)
from codewmbench.toolchains import inspect_local_toolchain
from codewmbench.watermarks.registry import available_watermarks
from codewmbench.watermarks.upstream_runtime import is_runtime_watermark

validate_stone_checkout = validate_runtime_checkout

from _shared import (
    CONFIG_DIR,
    DATA_DIR,
    DEFAULT_ATTACKS,
    DEFAULT_FIXTURE,
    DEFAULT_NORMALIZED_BENCHMARK,
    MODEL_CACHE_DIR,
    RESULTS_DIR,
    dump_json,
    load_config,
    load_json,
    read_jsonl,
)


IDENTITY_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "author": re.compile(r"(?im)^\s*author\s*:\s*.+$"),
    "affiliation": re.compile(r"(?im)^\s*affiliation\s*:\s*.+$"),
    "institution": re.compile(r"(?im)^\s*institution\s*:\s*.+$"),
    "personal_path": re.compile(r"(?i)c:\\users\\[a-z0-9._-]+"),
    "coordination_path": re.compile(r"(?i)(?:^|[\\/])\.coordination(?:[\\/]|$)"),
}

PUBLIC_BENCHMARK_CONTENT_PATTERN_LABELS = {
    "author",
    "affiliation",
    "institution",
    "personal_path",
    "coordination_path",
}

BENCHMARK_CONTENT_FIELDS = {
    "prompt",
    "reference_solution",
    "canonical_solution",
    "completion",
    "test",
    "execution_tests",
    "reference_tests",
    "contract",
    "expected_behavior",
    "semantic_contract",
    "stress_tests",
    "metamorphic_tests",
    "functional_cases",
    "stress_cases",
    "base_cases",
    "prompt_prefix",
}

BENCHMARK_METADATA_FIELDS = {
    "task_id",
    "dataset",
    "language",
    "source",
    "source_path",
    "source_url",
    "source_revision",
    "source_sha256",
    "source_digest",
    "prompt_digest",
    "solution_digest",
    "split",
    "license_note",
    "adapter_name",
    "validation_scope",
    "public_source",
    "record_kind",
    "source_group",
    "origin_type",
    "family_id",
    "difficulty",
    "evaluation_backend",
    "runner_image",
    "official_problem_file",
    "language_version",
    "reference_kind",
    "smoke_completion_available",
    "canonical_available",
    "notes",
    "description",
    "translation_anchor_language",
    "entry_point",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CodeWMBench setup.")
    parser.add_argument("--config", type=Path, default=Path("configs/debug.yaml"), help="Config file to validate.")
    parser.add_argument("--benchmark", type=Path, default=None, help="Canonical normalized benchmark fixture path.")
    parser.add_argument("--attack-matrix", type=Path, default=DEFAULT_ATTACKS, help="Attack matrix path.")
    parser.add_argument("--check-anonymity", action="store_true", help="Scan release-facing files for identity markers.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def ensure_paths_exist(paths: list[Path]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        if not path.exists():
            issues.append(f"missing: {path}")
    return issues


def scan_for_identity_markers(paths: list[Path], *, labels: set[str] | None = None) -> list[str]:
    findings: list[str] = []
    active_labels = labels or set(IDENTITY_PATTERNS)
    for path in paths:
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or "_cache" in path.parts or path.suffix.lower() in {".pyc", ".pyo", ".gz"}:
            continue
        if path.suffix.lower() in {".jsonl", ".json"} and "data" in path.parts:
            findings.extend(scan_for_identity_markers_in_benchmark(path))
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        findings.extend(_scan_text(text, path=path, labels=active_labels))
    return findings


def _scan_text(text: str, *, path: Path, labels: set[str] | None = None) -> list[str]:
    findings: list[str] = []
    active_labels = labels or set(IDENTITY_PATTERNS)
    for label, pattern in IDENTITY_PATTERNS.items():
        if label not in active_labels:
            continue
        if pattern.search(text):
            findings.append(f"{path}: matched {label}")
    return findings


def _collect_strings(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts: list[str] = []
        for nested in value.values():
            parts.extend(_collect_strings(nested))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            parts.extend(_collect_strings(item))
        return parts
    return [str(value)]


def scan_for_identity_markers_in_benchmark(path: Path) -> list[str]:
    findings: list[str] = []
    is_public_snapshot = "public" in path.parts
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            metadata_payload: list[str] = []
            content_payload: list[str] = []
            for key, value in row.items():
                key_str = str(key)
                if key_str in BENCHMARK_CONTENT_FIELDS:
                    content_payload.extend(_collect_strings(value))
                    continue
                if key_str not in BENCHMARK_METADATA_FIELDS and key_str not in {"notes", "description"}:
                    continue
                metadata_payload.extend(_collect_strings(value))
            if metadata_payload:
                findings.extend(_scan_text("\n".join(metadata_payload), path=Path(f"{path}#{index}")))
            if content_payload:
                labels = PUBLIC_BENCHMARK_CONTENT_PATTERN_LABELS if is_public_snapshot else None
                findings.extend(_scan_text("\n".join(content_payload), path=Path(f"{path}#{index}:content"), labels=labels))
        return findings
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _scan_text(text, path=path)


def release_candidate_files() -> list[Path]:
    candidates = [
        CONFIG_DIR,
        DATA_DIR / "fixtures",
        DATA_DIR / "public",
        Path("codewmbench"),
        MODEL_CACHE_DIR / "README.md",
        RESULTS_DIR / "schema.json",
        Path("docs"),
        Path("submission"),
        Path("README.md"),
        Path("Makefile"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path(".env.example"),
        Path(".gitignore"),
        Path("scripts"),
        Path("third_party"),
    ]
    return candidates


def _strict_local_hf_requirement(config: dict[str, Any], *, config_path: Path) -> HFModelRequirement | None:
    provider = dict(config.get("provider", {}))
    watermark = dict(config.get("watermark", {}))
    provider_mode = str(provider.get("mode", "")).strip().lower()
    provider_parameters = dict(provider.get("parameters", {}))
    watermark_scheme = str(watermark.get("scheme", "")).strip().lower()

    if provider_mode == "local_hf" and bool(provider_parameters.get("local_files_only", False)):
        model_name = str(provider_parameters.get("model", "")).strip()
        if not model_name:
            return None
        return HFModelRequirement(
            model=model_name,
            cache_dir=str(provider_parameters.get("cache_dir", "")).strip(),
            local_files_only=True,
            trust_remote_code=bool(provider_parameters.get("trust_remote_code", False)),
            device=str(provider_parameters.get("device", "cuda")).strip() or "cuda",
            dtype=str(provider_parameters.get("dtype", "float16")).strip() or "float16",
            token_env=str(provider_parameters.get("token_env", "HF_ACCESS_TOKEN")).strip() or "HF_ACCESS_TOKEN",
            usage=("local_hf",),
            config_paths=(str(config_path),),
        )

    if is_runtime_watermark(watermark_scheme) and bool(watermark.get("local_files_only", False)):
        model_name = str(watermark.get("model_name", "")).strip()
        if not model_name:
            return None
        return HFModelRequirement(
            model=model_name,
            cache_dir=str(watermark.get("cache_dir", "")).strip(),
            local_files_only=True,
            trust_remote_code=bool(watermark.get("trust_remote_code", False)),
            device=str(watermark.get("device", "cuda")).strip() or "cuda",
            dtype=str(watermark.get("dtype", "float16")).strip() or "float16",
            token_env=str(watermark.get("token_env", "HF_ACCESS_TOKEN")).strip() or "HF_ACCESS_TOKEN",
            usage=("runtime",),
            config_paths=(str(config_path),),
        )

    return None


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    benchmark_path = args.benchmark
    if benchmark_path is None:
        benchmark_cfg = config.get("benchmark", {})
        path_cfg = config.get("paths", {})
        benchmark_path = Path(
            benchmark_cfg.get("prepared_output")
            or path_cfg.get("prepared_benchmark")
            or DEFAULT_NORMALIZED_BENCHMARK
        )
    benchmark_rows = read_jsonl(benchmark_path)
    attack_matrix = load_json(args.attack_matrix)
    benchmark_manifest_path = benchmark_path.with_suffix(".manifest.json")
    benchmark_manifest = load_json(benchmark_manifest_path) if benchmark_manifest_path.exists() else {}
    known_attacks = set(available_attacks())
    known_watermarks = set(available_watermarks())
    runtime_requirements: dict[str, object] = {}
    hf_cache_validation: dict[str, object] | None = None
    baseline_provenance: dict[str, Any] = {}
    toolchain_validation: dict[str, Any] = {}

    issues = []
    issues.extend(ensure_paths_exist([args.config, benchmark_path, args.attack_matrix, CONFIG_DIR, RESULTS_DIR]))
    if not benchmark_manifest:
        issues.append(f"missing benchmark manifest: {benchmark_manifest_path}")
    if not benchmark_rows:
        issues.append("benchmark fixture is empty")
    if benchmark_manifest and benchmark_manifest.get("record_count") not in {None, len(benchmark_rows)}:
        issues.append(
            f"benchmark manifest count mismatch: manifest={benchmark_manifest.get('record_count')} rows={len(benchmark_rows)}"
        )
    if not isinstance(attack_matrix.get("attacks"), list) or not attack_matrix["attacks"]:
        issues.append("attack matrix is empty")
    watermark_config = config.get("watermark", {})
    watermark_scheme = str(watermark_config.get("scheme", "")).lower()
    if watermark_scheme not in known_watermarks:
        issues.append(
            f"unknown watermark scheme: {watermark_config.get('scheme', '')}"
        )
    if is_runtime_watermark(watermark_scheme):
        model_name = str(watermark_config.get("model_name", "")).strip()
        baseline_provenance = stone_family_checkout_metadata(watermark_scheme)
        upstream_root = Path(str(baseline_provenance["repo_root"])) if baseline_provenance.get("repo_root") else None
        runtime_requirements = {
            "scheme": watermark_scheme,
            "model_name": model_name,
            "upstream_root": str(upstream_root) if upstream_root is not None else None,
            "token_env": str(watermark_config.get("token_env", "HF_ACCESS_TOKEN")),
            "device": str(watermark_config.get("device", "auto")),
            "baseline_provenance": baseline_provenance,
        }
        if not model_name:
            issues.append(f"runtime watermark '{watermark_scheme}' requires watermark.model_name")
        issues.extend(validate_runtime_checkout(watermark_scheme))
        if upstream_root is None:
            issues.append(
                f"runtime watermark '{watermark_scheme}' requires a valid official upstream checkout"
            )
    configured_attacks = [str(name).lower() for name in config.get("attacks", {}).get("include", [])]
    unknown_attacks = sorted(set(configured_attacks) - known_attacks)
    if unknown_attacks:
        issues.append(f"unknown configured attacks: {unknown_attacks}")
    matrix_names = [str(item.get("name", "")).lower() for item in attack_matrix.get("attacks", []) if isinstance(item, dict)]
    unknown_matrix_names = sorted(set(matrix_names) - known_attacks)
    if unknown_matrix_names:
        issues.append(f"attack matrix contains unknown attacks: {unknown_matrix_names}")
    benchmark_languages = [str(language).lower() for language in config.get("benchmark", {}).get("languages", [])]
    if benchmark_manifest:
        observed_languages = [str(language).lower() for language in benchmark_manifest.get("observed_languages", [])]
        missing_claimed_languages = [language for language in benchmark_languages if language not in observed_languages]
        if missing_claimed_languages:
            issues.append(f"claimed benchmark languages missing from fixture: {missing_claimed_languages}")
    runtime_validation_languages = sorted(
        {
            str(row.get("language", "")).strip().lower()
            for row in benchmark_rows
            if str(row.get("evaluation_backend", "")).strip().lower() == "docker_remote"
            and bool(row.get("validation_supported"))
        }
    )
    if runtime_validation_languages:
        toolchain_entries: list[dict[str, Any]] = []
        for language in runtime_validation_languages:
            inspection = inspect_local_toolchain(language)
            toolchain_entries.append(inspection)
            if inspection.get("status") != "ok":
                issues.extend(
                    f"toolchain[{language}] {item}"
                    for item in inspection.get("issues", [])
                )
        toolchain_validation = {
            "languages": runtime_validation_languages,
            "entries": toolchain_entries,
        }

    strict_hf_requirement = _strict_local_hf_requirement(config, config_path=args.config)
    if strict_hf_requirement is not None:
        hf_cache_validation = validate_local_hf_cache(strict_hf_requirement, require_root_entry=True)
        runtime_requirements["hf_model_requirement"] = {
            "model": strict_hf_requirement.model,
            "cache_dir": strict_hf_requirement.cache_dir,
            "local_files_only": strict_hf_requirement.local_files_only,
            "usage": list(strict_hf_requirement.usage),
            "config_paths": list(strict_hf_requirement.config_paths),
        }
        if hf_cache_validation["status"] != "ok":
            issues.extend(str(item) for item in hf_cache_validation.get("issues", []))

    if args.check_anonymity:
        paths = []
        for candidate in release_candidate_files():
            if candidate.is_dir():
                paths.extend(sorted(candidate.rglob("*")))
            else:
                paths.append(candidate)
        findings = scan_for_identity_markers(paths, labels=set(IDENTITY_PATTERNS) - {"coordination_path"})
        issues.extend(findings)

    report = {
        "config_path": str(args.config),
        "benchmark_path": str(benchmark_path),
        "attack_matrix_path": str(args.attack_matrix),
        "config_keys": sorted(config.keys()),
        "benchmark_count": len(benchmark_rows),
        "attack_count": len(attack_matrix.get("attacks", [])),
        "benchmark_manifest": benchmark_manifest,
        "baseline_provenance": baseline_provenance,
        "runtime_requirements": runtime_requirements,
        "hf_cache_validation": hf_cache_validation,
        "toolchain_validation": toolchain_validation,
        "issues": issues,
    }

    if args.report is not None:
        dump_json(args.report, report)

    if issues:
        print("Validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Validation passed: setup is ready for local reproduction and public release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
