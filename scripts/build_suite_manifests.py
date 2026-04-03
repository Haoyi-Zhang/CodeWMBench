from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from codewmbench.config import load_config
from codewmbench.suite import (
    OFFICIAL_RUNTIME_BASELINES,
    OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
    SUITE_ATOMIC_SOURCE_LABELS,
    SUITE_ATOMIC_SOURCE_ORDER,
    SUITE_MODEL_ROSTER,
    SUITE_MODEL_SLUGS,
    suite_benchmark_roster,
    suite_experiment_languages,
    suite_source_by_slug,
)
from codewmbench.utils import stable_hash

try:
    from _shared import dump_json, read_jsonl, write_jsonl
except ModuleNotFoundError:  # pragma: no cover
    from scripts._shared import dump_json, read_jsonl, write_jsonl


HEAVY_STAGE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
MODEL_PRIORITY_ORDER = (
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "bigcode/starcoder2-7b",
)
METHOD_PRIORITY_ORDER = (
    "ewd_runtime",
    "kgw_runtime",
    "sweet_runtime",
    "stone_runtime",
)
SOURCE_PRIORITY_ORDER = (
    "crafted_original",
    "crafted_translation",
    "crafted_stress",
    "mbxp_5lang",
    "humaneval_x",
    "mbpp_plus",
    "humaneval_plus",
)
_MODEL_PRIORITY_RANK = {name: len(MODEL_PRIORITY_ORDER) - index for index, name in enumerate(MODEL_PRIORITY_ORDER)}
_METHOD_PRIORITY_RANK = {name: len(METHOD_PRIORITY_ORDER) - index for index, name in enumerate(METHOD_PRIORITY_ORDER)}
_SOURCE_PRIORITY_RANK = {name: len(SOURCE_PRIORITY_ORDER) - index for index, name in enumerate(SOURCE_PRIORITY_ORDER)}

_SOURCE_CONFIGS = {
    "humaneval_plus": "configs/public_humaneval_plus_stone_runtime.yaml",
    "mbpp_plus": "configs/public_mbpp_plus_stone_runtime.yaml",
    "humaneval_x": "configs/humanevalx_only.yaml",
    "mbxp_5lang": "configs/mbxp_only.yaml",
    "crafted_original": "configs/crafted_original_only.yaml",
    "crafted_translation": "configs/crafted_translation_only.yaml",
    "crafted_stress": "configs/crafted_stress_only.yaml",
}

_BASE_CONFIGS = {
    "stone_runtime": {
        "humaneval_plus": "configs/public_humaneval_plus_stone_runtime.yaml",
        "mbpp_plus": "configs/public_mbpp_plus_stone_runtime.yaml",
        "default": "configs/public_humaneval_plus_stone_runtime.yaml",
    },
    "sweet_runtime": {
        "humaneval_plus": "configs/public_humaneval_plus_sweet_runtime.yaml",
        "mbpp_plus": "configs/public_mbpp_plus_sweet_runtime.yaml",
        "default": "configs/public_humaneval_plus_sweet_runtime.yaml",
    },
    "ewd_runtime": {
        "humaneval_plus": "configs/public_humaneval_plus_ewd_runtime.yaml",
        "mbpp_plus": "configs/public_mbpp_plus_ewd_runtime.yaml",
        "default": "configs/public_humaneval_plus_ewd_runtime.yaml",
    },
    "kgw_runtime": {
        "humaneval_plus": "configs/public_humaneval_plus_kgw_runtime.yaml",
        "mbpp_plus": "configs/public_mbpp_plus_kgw_runtime.yaml",
        "default": "configs/public_humaneval_plus_kgw_runtime.yaml",
    },
}

_SOURCE_TAGS = {
    "humaneval_plus": "heplus",
    "mbpp_plus": "mbppplus",
    "humaneval_x": "humanevalx",
    "mbxp_5lang": "mbxp",
    "crafted_original": "crafted_original",
    "crafted_translation": "crafted_translation",
    "crafted_stress": "crafted_stress",
}

_COMPACT_INPUT_PATHS = {
    "mbpp_plus": ROOT / "data" / "public" / "mbpp_plus" / "normalized.jsonl",
    "humaneval_x": ROOT / "data" / "public" / "humaneval_x" / "normalized.jsonl",
    "mbxp_5lang": ROOT / "data" / "public" / "mbxp_5lang" / "normalized.jsonl",
    "crafted_original": ROOT / "data" / "compact" / "crafted" / "crafted_original.normalized.jsonl",
    "crafted_translation": ROOT / "data" / "compact" / "crafted" / "crafted_translation.normalized.jsonl",
    "crafted_stress": ROOT / "data" / "compact" / "crafted" / "crafted_stress.normalized.jsonl",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relpath(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _config_payload(path: str) -> dict[str, Any]:
    return dict(load_config(ROOT / path).raw)


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _stable_row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("family_id", "")).strip().lower(),
        str(row.get("task_id", "")).strip().lower(),
        _normalize(row.get("language")),
        _normalize(row.get("difficulty")),
        str(row.get("prompt_digest", "")).strip().lower(),
        str(row.get("source_digest", "")).strip().lower(),
    )


def _drain_round_robin(
    buckets: dict[object, list[dict[str, Any]]],
    *,
    ordered_keys: list[object],
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(buckets.get(key) for key in ordered_keys):
        for key in ordered_keys:
            bucket = buckets.get(key) or []
            if not bucket:
                continue
            selected.append(dict(bucket.pop(0)))
            if len(selected) >= limit:
                break
    return selected


def _difficulty_round_robin(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered_difficulties = ["hard", "medium", "easy"]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_normalize(row.get("difficulty"))].append(dict(row))
    for bucket in buckets.values():
        bucket.sort(key=_stable_row_sort_key)
    return _drain_round_robin(buckets, ordered_keys=ordered_difficulties, limit=limit)


def _candidate_multilingual_families(
    rows: list[dict[str, Any]],
    *,
    languages: tuple[str, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_family: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        family_id = str(row.get("family_id", "")).strip()
        language = _normalize(row.get("language"))
        if not family_id or language not in languages:
            continue
        current = by_family[family_id].get(language)
        candidate = dict(row)
        if current is None or _stable_row_sort_key(candidate) < _stable_row_sort_key(current):
            by_family[family_id][language] = candidate
    return {
        family_id: bundle
        for family_id, bundle in by_family.items()
        if all(language in bundle for language in languages)
    }


def _family_round_robin_selection(
    family_rows: dict[str, dict[str, dict[str, Any]]],
    *,
    languages: tuple[str, ...],
    family_limit: int,
    stratum_fields: tuple[str, ...],
) -> list[str]:
    buckets: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for family_id, bundle in family_rows.items():
        representative = bundle[languages[0]]
        key = tuple(_normalize(representative.get(field)) or "unknown" for field in stratum_fields)
        buckets[key].append(family_id)
    ordered_keys = sorted(buckets, key=lambda key: tuple(str(part) for part in key))
    for bucket in buckets.values():
        bucket.sort()
    selected: list[str] = []
    while len(selected) < family_limit and any(buckets.get(key) for key in ordered_keys):
        for key in ordered_keys:
            bucket = buckets.get(key) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= family_limit:
                break
    return selected


def _category_balanced_family_selection(
    family_rows: dict[str, dict[str, dict[str, Any]]],
    *,
    languages: tuple[str, ...],
    family_limit: int,
    difficulty_field: str = "difficulty",
) -> list[str]:
    category_buckets: dict[str, list[str]] = defaultdict(list)
    representative_rows: dict[str, dict[str, Any]] = {}
    for family_id, bundle in family_rows.items():
        representative = bundle[languages[0]]
        representative_rows[family_id] = representative
        category_buckets[_normalize(representative.get("category")) or "unknown"].append(family_id)

    ordered_categories = sorted(category_buckets)
    if not ordered_categories:
        return []
    base = family_limit // len(ordered_categories)
    remainder = family_limit % len(ordered_categories)
    quotas = {category: base for category in ordered_categories}
    availability_order = sorted(
        ordered_categories,
        key=lambda category: (-len(category_buckets[category]), category),
    )
    for category in availability_order[:remainder]:
        quotas[category] += 1

    selected: list[str] = []
    for category in ordered_categories:
        family_ids = category_buckets[category]
        difficulty_buckets: dict[str, list[str]] = defaultdict(list)
        for family_id in family_ids:
            representative = representative_rows[family_id]
            difficulty_buckets[_normalize(representative.get(difficulty_field)) or "unknown"].append(family_id)
        for bucket in difficulty_buckets.values():
            bucket.sort()
        category_selected: list[str] = []
        ordered_difficulties = ["hard", "medium", "easy", "unknown"]
        while len(category_selected) < min(int(quotas[category]), len(family_ids)) and any(
            difficulty_buckets.get(key) for key in ordered_difficulties
        ):
            for difficulty in ordered_difficulties:
                bucket = difficulty_buckets.get(difficulty) or []
                if not bucket:
                    continue
                category_selected.append(bucket.pop(0))
                if len(category_selected) >= min(int(quotas[category]), len(family_ids)):
                    break
        selected.extend(category_selected)

    if len(selected) < family_limit:
        remaining = sorted(set(family_rows) - set(selected))
        selected.extend(remaining[: family_limit - len(selected)])
    return selected[:family_limit]


def _selected_family_rows(
    family_rows: dict[str, dict[str, dict[str, Any]]],
    *,
    selected_family_ids: list[str],
    languages: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for family_id in selected_family_ids:
        bundle = family_rows[family_id]
        for language in languages:
            selected.append(dict(bundle[language]))
    return selected


def _coverage_from_rows(rows: list[dict[str, Any]], *, claimed_languages: tuple[str, ...]) -> dict[str, Any]:
    languages = sorted({_normalize(row.get("language")) for row in rows if _normalize(row.get("language"))})
    validation_supported_languages = sorted(
        {
            _normalize(row.get("language"))
            for row in rows
            if _normalize(row.get("language")) and bool(row.get("validation_supported"))
        }
    )
    runtime_annotation_available = any("clean_reference_validation_available" in row for row in rows)
    compile_annotation_available = any("clean_reference_compile_success" in row for row in rows)
    pass_annotation_available = any("clean_reference_passed" in row for row in rows)
    runtime_validation_supported_languages = (
        sorted(
            {
                _normalize(row.get("language"))
                for row in rows
                if _normalize(row.get("language")) and bool(row.get("clean_reference_validation_available"))
            }
        )
        if runtime_annotation_available
        else []
    )
    clean_reference_compile_rate = (
        round(sum(1 for row in rows if row.get("clean_reference_compile_success") is True) / max(1, len(rows)), 4)
        if compile_annotation_available
        else None
    )
    clean_reference_pass_rate = (
        round(sum(1 for row in rows if row.get("clean_reference_passed") is True) / max(1, len(rows)), 4)
        if pass_annotation_available
        else None
    )
    claimed = list(claimed_languages) or languages
    return {
        "observed_language_count": len(languages),
        "claimed_language_count": len(claimed),
        "observed_coverage_rate": round(len(set(languages) & set(claimed)) / max(1, len(claimed)), 4),
        "declared_semantic_validation_rate": round(sum(1 for row in rows if bool(row.get("validation_supported"))) / max(1, len(rows)), 4),
        "declared_semantic_validation_language_rate": round(len(set(validation_supported_languages)) / max(1, len(claimed)), 4),
        "runtime_semantic_validation_rate": (
            round(sum(1 for row in rows if row.get("clean_reference_validation_available")) / max(1, len(rows)), 4)
            if runtime_annotation_available
            else None
        ),
        "runtime_semantic_validation_language_rate": (
            round(len(set(runtime_validation_supported_languages)) / max(1, len(claimed)), 4)
            if runtime_annotation_available
            else None
        ),
        "semantic_validation_rate": (
            round(sum(1 for row in rows if row.get("clean_reference_validation_available")) / max(1, len(rows)), 4)
            if runtime_annotation_available
            else None
        ),
        "semantic_validation_language_rate": (
            round(len(set(runtime_validation_supported_languages)) / max(1, len(claimed)), 4)
            if runtime_annotation_available
            else None
        ),
        "clean_reference_compile_rate": clean_reference_compile_rate,
        "clean_reference_pass_rate": clean_reference_pass_rate,
        "runtime_validation_basis": "row_annotations" if runtime_annotation_available else "unavailable",
        "runtime_validation_annotations_available": runtime_annotation_available,
        "missing_claimed_languages": sorted(set(claimed) - set(languages)),
        "declared_unvalidated_languages": sorted(
            {
                _normalize(row.get("language"))
                for row in rows
                if _normalize(row.get("language")) and not bool(row.get("validation_supported"))
            }
        ),
        "runtime_unvalidated_languages": (
            sorted(
                {
                    _normalize(row.get("language"))
                    for row in rows
                    if _normalize(row.get("language")) and not bool(row.get("clean_reference_validation_available"))
                }
            )
            if runtime_annotation_available
            else []
        ),
        "unvalidated_languages": (
            sorted(
                {
                    _normalize(row.get("language"))
                    for row in rows
                    if _normalize(row.get("language")) and not bool(row.get("clean_reference_validation_available"))
                }
            )
            if runtime_annotation_available
            else []
        ),
    }


def _source_manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("source_manifests")
    if isinstance(entries, list) and entries:
        return [dict(item) for item in entries]
    return [dict(manifest)]


def _write_compact_collection(
    *,
    source_slug: str,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    input_candidate_count: int,
    claimed_languages: tuple[str, ...],
    selection_policy: dict[str, Any],
) -> None:
    base_manifest = _load_json(input_path.with_suffix(".manifest.json"))
    spec = suite_source_by_slug(source_slug)
    if spec is None:
        raise KeyError(f"Unknown suite source slug: {source_slug}")
    rows = sorted((dict(row) for row in rows), key=_stable_row_sort_key)
    write_jsonl(output_path, rows)
    language_counts = Counter(_normalize(row.get("language")) for row in rows if _normalize(row.get("language")))
    source_group_counts = Counter(_normalize(row.get("source_group")) for row in rows if _normalize(row.get("source_group")))
    origin_type_counts = Counter(_normalize(row.get("origin_type")) for row in rows if _normalize(row.get("origin_type")))
    difficulty_counts = Counter(_normalize(row.get("difficulty")) for row in rows if _normalize(row.get("difficulty")))
    reference_kind_counts = Counter(_normalize(row.get("reference_kind") or "canonical") for row in rows)
    family_counts = Counter(str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip())
    category_counts = Counter(_normalize(row.get("category")) for row in rows if _normalize(row.get("category")))
    template_family_counts = Counter(_normalize(row.get("template_family")) for row in rows if _normalize(row.get("template_family")))
    sample_ids_path = output_path.with_suffix(".sample_ids.json")
    dump_json(sample_ids_path, {"record_count": len(rows), "sample_ids": [str(row.get("task_id", "")).strip() for row in rows if str(row.get("task_id", "")).strip()]})
    manifest = {
        "schema_version": 1,
        "collection_name": spec.collection_name or output_path.stem.replace(".normalized", ""),
        "benchmark": str(base_manifest.get("benchmark") or spec.dataset_label),
        "dataset_label": str(spec.dataset_label),
        "record_count": len(rows),
        "task_count": len(rows),
        "observed_languages": sorted(language_counts),
        "claimed_languages": list(claimed_languages),
        "validation_supported_languages": sorted({_normalize(row.get("language")) for row in rows if _normalize(row.get("language")) and bool(row.get("validation_supported"))}),
        "datasets": sorted({str(row.get("dataset", "")).strip() for row in rows if str(row.get("dataset", "")).strip()}),
        "language_counts": dict(sorted(language_counts.items())),
        "source_group_counts": dict(sorted(source_group_counts.items())),
        "origin_type_counts": dict(sorted(origin_type_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "family_count": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "inputs": [_relpath(input_path)],
        "input_filtered_counts": {_relpath(input_path): int(input_candidate_count)},
        "include_languages": list(claimed_languages),
        "include_source_groups": sorted(source_group_counts),
        "include_origin_types": sorted(origin_type_counts),
        "include_difficulties": sorted(difficulty_counts),
        "include_reference_kinds": sorted(reference_kind_counts),
        "quota_per_language": dict(sorted(language_counts.items())),
        "quota_per_source_group": {},
        "reference_kind_counts": dict(sorted(reference_kind_counts.items())),
        "reference_kind_total": int(sum(reference_kind_counts.values())),
        "canonical_reference_count": int(reference_kind_counts.get("canonical", 0)),
        "smoke_overlay_reference_count": int(reference_kind_counts.get("smoke_overlay", 0)),
        "coverage": _coverage_from_rows(rows, claimed_languages=claimed_languages),
        "source_manifests": _source_manifest_entries(base_manifest),
        "sample_ids_path": _relpath(sample_ids_path),
        "suite_selection_policy": selection_policy,
    }
    if category_counts:
        manifest["category_counts"] = dict(sorted(category_counts.items()))
    if template_family_counts:
        manifest["template_family_counts"] = dict(sorted(template_family_counts.items()))
    dump_json(output_path.with_suffix(".manifest.json"), manifest)


def _build_mbpp_plus_compact() -> None:
    spec = suite_source_by_slug("mbpp_plus")
    if spec is None:
        raise KeyError("Unknown suite source slug: mbpp_plus")
    input_path = _COMPACT_INPUT_PATHS["mbpp_plus"]
    output_path = ROOT / spec.prepared_output
    rows = read_jsonl(input_path)
    selected = _difficulty_round_robin(rows, limit=int(spec.full_limit))
    _write_compact_collection(
        source_slug="mbpp_plus",
        input_path=input_path,
        output_path=output_path,
        rows=selected,
        input_candidate_count=len(rows),
        claimed_languages=("python",),
        selection_policy={
            "type": "difficulty_round_robin",
            "target_record_count": int(spec.full_limit),
            "difficulty_order": ["hard", "medium", "easy"],
            "family_balance": "implicit_one_task_per_family",
        },
    )


def _build_multilingual_family_compact(
    source_slug: str,
    *,
    stratum_fields: tuple[str, ...],
    category_balanced: bool = False,
) -> None:
    spec = suite_source_by_slug(source_slug)
    if spec is None:
        raise KeyError(f"Unknown suite source slug: {source_slug}")
    input_path = _COMPACT_INPUT_PATHS[source_slug]
    output_path = ROOT / spec.prepared_output
    rows = read_jsonl(input_path)
    languages = tuple(suite_experiment_languages(spec))
    family_rows = _candidate_multilingual_families(rows, languages=languages)
    family_limit = int(spec.full_limit) // max(1, len(languages))
    if category_balanced:
        selected_family_ids = _category_balanced_family_selection(
            family_rows,
            languages=languages,
            family_limit=family_limit,
        )
    else:
        selected_family_ids = _family_round_robin_selection(
            family_rows,
            languages=languages,
            family_limit=family_limit,
            stratum_fields=stratum_fields,
        )
    selected_rows = _selected_family_rows(family_rows, selected_family_ids=selected_family_ids, languages=languages)
    _write_compact_collection(
        source_slug=source_slug,
        input_path=input_path,
        output_path=output_path,
        rows=selected_rows,
        input_candidate_count=len(family_rows) * len(languages),
        claimed_languages=languages,
        selection_policy={
            "type": "category_balanced_multilingual_compact_slice" if category_balanced else "family_balanced_multilingual_compact_slice",
            "target_record_count": int(spec.full_limit),
            "target_family_count": family_limit,
            "languages": list(languages),
            "strata": list(stratum_fields),
            "common_support_languages": list(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        },
    )


def _build_compact_prepared_collections() -> None:
    _build_mbpp_plus_compact()
    _build_multilingual_family_compact("humaneval_x", stratum_fields=("difficulty",))
    _build_multilingual_family_compact("mbxp_5lang", stratum_fields=("difficulty",))
    _build_multilingual_family_compact("crafted_original", stratum_fields=("category", "difficulty"), category_balanced=True)
    _build_multilingual_family_compact("crafted_translation", stratum_fields=("category", "difficulty"), category_balanced=True)
    _build_multilingual_family_compact("crafted_stress", stratum_fields=("category", "difficulty"), category_balanced=True)


def _source_metadata(source_key: str) -> dict[str, Any]:
    config_path = _SOURCE_CONFIGS[source_key]
    payload = _config_payload(config_path)
    benchmark = dict(payload.get("benchmark", {}))
    paths = dict(payload.get("paths", {}))
    source_spec = suite_source_by_slug(source_key)
    if source_spec is None:
        raise KeyError(f"Unknown suite source slug: {source_key}")
    prepared = str(
        source_spec.prepared_output
        or source_spec.prepared_benchmark
        or benchmark.get("prepared_output")
        or paths.get("prepared_benchmark")
        or benchmark.get("source")
        or ""
    ).strip()
    manifest = _load_json((ROOT / prepared).with_suffix(".manifest.json"))
    dataset_label = str(source_spec.dataset_label).strip() or SUITE_ATOMIC_SOURCE_LABELS[source_key]
    source_group_counts = dict(manifest.get("source_group_counts", {}))
    source_group = str(source_spec.source_group or benchmark.get("source_group") or next(iter(source_group_counts.keys()), source_key)).strip()
    observed_languages = [str(language).strip() for language in manifest.get("observed_languages", []) if str(language).strip()]
    experiment_languages = [str(language).strip() for language in suite_experiment_languages(source_spec) if str(language).strip()]
    if observed_languages:
        observed_languages = [language for language in observed_languages if language in experiment_languages]
    if not observed_languages:
        observed_languages = experiment_languages
    benchmark_override = dict(benchmark)
    benchmark_override["dataset_label"] = dataset_label
    benchmark_override["source"] = prepared
    benchmark_override["prepared_output"] = prepared
    benchmark_override["languages"] = observed_languages
    benchmark_override["source_group"] = source_group
    benchmark_override.pop("collection_sources", None)
    if source_spec.collection_name:
        benchmark_override["collection_name"] = source_spec.collection_name
    return {
        "key": source_key,
        "config": config_path,
        "dataset_label": dataset_label,
        "prepared_benchmark": prepared,
        "benchmark": benchmark_override,
        "full_limit": int(source_spec.full_limit),
        "source_group": source_group,
        "languages": observed_languages,
        "tag": _SOURCE_TAGS[source_key],
    }


def _base_config(method: str, source_key: str) -> str:
    spec = _BASE_CONFIGS[method]
    return spec.get(source_key, spec["default"])


def _project_name(method: str, model: str, source_key: str, stage: str) -> str:
    model_slug = SUITE_MODEL_SLUGS[model]
    return f"codewmbench-{stage}-{method}-{model_slug}-{_SOURCE_TAGS[source_key]}"


def _shared_seed(*, model: str, source_key: str, stage: str) -> int:
    digest = stable_hash(f"{stage}:{model}:{source_key}", digest_size=8)
    return 1000 + (int(digest, 16) % 900_000)


def _estimated_priority(*, model: str, method: str, source_key: str, benchmark_limit: int) -> int:
    model_rank = int(_MODEL_PRIORITY_RANK.get(model, 0))
    method_rank = int(_METHOD_PRIORITY_RANK.get(method, 0))
    source_rank = int(_SOURCE_PRIORITY_RANK.get(source_key, 0))
    capped_limit = max(0, min(int(benchmark_limit), 99_999))
    return model_rank * 1_000_000_000 + method_rank * 10_000_000 + source_rank * 100_000 + capped_limit


def _run_item(
    *,
    run_id: str,
    profile: str,
    config: str,
    model: str,
    source: dict[str, Any],
    stage: str,
    benchmark_limit: int | None,
    baseline_eval_sample_limit: int,
    tags: list[str],
) -> dict[str, Any]:
    if benchmark_limit is None:
        raise ValueError(f"{run_id} requires an explicit benchmark limit for heavy-first priority planning")
    benchmark_override = json.loads(json.dumps(source["benchmark"]))
    benchmark_override["limit"] = int(benchmark_limit)
    return {
        "run_id": run_id,
        "profile": profile,
        "config": config,
        "config_overrides": {
            "project": {"name": _project_name(tags[-1], model, source["key"], stage), "seed": _shared_seed(model=model, source_key=source["key"], stage=stage)},
            "paths": {"prepared_benchmark": source["prepared_benchmark"]},
            "benchmark": benchmark_override,
            "watermark": {"model_name": model},
        },
        "resource": "gpu",
        "gpu_pool": "runtime",
        "baseline_eval": True,
        "baseline_eval_sample_limit": int(baseline_eval_sample_limit),
        "priority": _estimated_priority(model=model, method=tags[-1], source_key=source["key"], benchmark_limit=int(benchmark_limit)),
        "tags": tags,
    }


def _suite_run_items() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    source_specs = [_source_metadata(source_key) for source_key in SUITE_ATOMIC_SOURCE_ORDER]
    for model in SUITE_MODEL_ROSTER:
        model_slug = SUITE_MODEL_SLUGS[model]
        for source in source_specs:
            for method in OFFICIAL_RUNTIME_BASELINES:
                runs.append(
                    _run_item(
                        run_id=f"suite_{model_slug}_{source['tag']}_{method}",
                        profile="suite_all_models_methods",
                        config=_base_config(method, source["key"]),
                        model=model,
                        source=source,
                        stage="suite",
                        benchmark_limit=int(source["full_limit"]),
                        baseline_eval_sample_limit=64,
                        tags=["suite", model_slug, source["tag"], method, method],
                    )
                )
    return runs


def _stage_a_run_items() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    model = HEAVY_STAGE_MODEL
    model_slug = SUITE_MODEL_SLUGS[model]
    for source_key in SUITE_ATOMIC_SOURCE_ORDER:
        source = _source_metadata(source_key)
        source_spec = suite_source_by_slug(source_key)
        if source_spec is None:
            raise KeyError(f"Unknown suite source slug: {source_key}")
        for method in OFFICIAL_RUNTIME_BASELINES:
            runs.append(
                _run_item(
                    run_id=f"stage_a_{model_slug}_{source['tag']}_{method}",
                    profile="suite_canary_heavy",
                    config=_base_config(method, source["key"]),
                    model=model,
                    source=source,
                    stage="stage-a",
                    benchmark_limit=int(source_spec.stage_a_limit),
                    baseline_eval_sample_limit=16,
                    tags=["precheck", "stage_a", model_slug, source["tag"], method, method],
                )
            )
    return runs


def _stage_b_run_items() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    smoke_sources: list[tuple[dict[str, Any], int]] = []
    for source_key in ("humaneval_plus", "mbpp_plus"):
        source = _source_metadata(source_key)
        source_spec = suite_source_by_slug(source_key)
        if source_spec is None:
            raise KeyError(f"Unknown suite source slug: {source_key}")
        smoke_sources.append((source, int(source_spec.stage_b_limit)))
    for model in SUITE_MODEL_ROSTER:
        if model == HEAVY_STAGE_MODEL:
            continue
        model_slug = SUITE_MODEL_SLUGS[model]
        for source, stage_b_limit in smoke_sources:
            for method in OFFICIAL_RUNTIME_BASELINES:
                runs.append(
                    _run_item(
                        run_id=f"stage_b_{model_slug}_{source['tag']}_{method}",
                        profile="model_invocation_smoke",
                        config=_base_config(method, source["key"]),
                        model=model,
                        source=source,
                        stage="stage-b",
                        benchmark_limit=stage_b_limit,
                        baseline_eval_sample_limit=4,
                        tags=["precheck", "stage_b", model_slug, source["tag"], method, method],
                    )
                )
    return runs


def _manifest_payload(
    *,
    profile: str,
    description: str,
    runs: list[dict[str, Any]],
    model_roster: list[str] | tuple[str, ...],
    benchmark_roster: list[str] | tuple[str, ...],
    atomic_benchmark_sources: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "profile": profile,
        "description": description,
        "fairness_rule": "For each local model backbone, compare all four official imported baselines on the same executable benchmark rows; cross-model execution may run in parallel but aggregation remains model-conditioned.",
        "model_roster": list(model_roster),
        "benchmark_roster": list(benchmark_roster),
        "atomic_benchmark_sources": list(atomic_benchmark_sources),
        "suite_inventory_datasets": ["HumanEval", *suite_benchmark_roster()],
        "method_roster": list(OFFICIAL_RUNTIME_BASELINES),
        "required_watermark_methods": list(OFFICIAL_RUNTIME_BASELINES),
        "required_provider_modes": ["offline_mock"],
        "required_gpu_pools": ["runtime"],
        "runs": runs,
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    _build_compact_prepared_collections()
    manifests = {
        ROOT / "configs" / "matrices" / "suite_all_models_methods.json": _manifest_payload(
            profile="suite_all_models_methods",
            description="TOSEM-compact CodeWMBench full matrix: 4 local backbones x 4 pinned baseline implementations across 7 compact atomic benchmark sources.",
            runs=_suite_run_items(),
            model_roster=SUITE_MODEL_ROSTER,
            benchmark_roster=suite_benchmark_roster(),
            atomic_benchmark_sources=SUITE_ATOMIC_SOURCE_ORDER,
        ),
        ROOT / "configs" / "matrices" / "suite_canary_heavy.json": _manifest_payload(
            profile="suite_canary_heavy",
            description="Stage A heavy precheck: Qwen2.5-Coder-14B across the compact CodeWMBench atomic-source suite with bounded sample counts.",
            runs=_stage_a_run_items(),
            model_roster=(HEAVY_STAGE_MODEL,),
            benchmark_roster=suite_benchmark_roster(),
            atomic_benchmark_sources=SUITE_ATOMIC_SOURCE_ORDER,
        ),
        ROOT / "configs" / "matrices" / "model_invocation_smoke.json": _manifest_payload(
            profile="model_invocation_smoke",
            description="Stage B smoke precheck: remaining three backbones across HumanEval+ and compact MBPP+ with bounded sample counts.",
            runs=_stage_b_run_items(),
            model_roster=tuple(model for model in SUITE_MODEL_ROSTER if model != HEAVY_STAGE_MODEL),
            benchmark_roster=("HumanEval+", "MBPP+"),
            atomic_benchmark_sources=("humaneval_plus", "mbpp_plus"),
        ),
    }
    for path, payload in manifests.items():
        _write_manifest(path, payload)
    print(json.dumps({str(path.relative_to(ROOT)): len(payload["runs"]) for path, payload in manifests.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
