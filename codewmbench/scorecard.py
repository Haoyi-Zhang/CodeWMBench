from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import mean
from typing import Any, Iterable, Mapping

from .baselines.stone_family.evaluation import binary_auroc
from .models import BenchmarkRow
from .suite import normalize_source_group


EPSILON = 1e-9
SPARSE_TASK_MIN_ROWS = 20
NEGATIVE_CONTROL_FAMILIES = ("human_reference", "clean_generation")
SOURCE_BALANCED_COMPONENT_FIELDS = (
    "detection_reliability",
    "robustness",
    "utility",
    "stealth",
    "slice_core",
    "negative_control_fpr",
    "negative_vs_watermarked_auroc",
    "negative_control_support_rate",
    "semantic_validation_rate",
    "declared_semantic_validation_rate",
    "watermarked_pass_preservation",
    "attacked_pass_preservation",
)


def _clamp01(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0 if numerator <= 0 else 1.0
    return _clamp01(float(numerator) / max(float(denominator), EPSILON))


def _row_metadata(row: BenchmarkRow) -> dict[str, Any]:
    return dict(row.metadata) if isinstance(row.metadata, Mapping) else {}


def _example_metadata(row: BenchmarkRow) -> dict[str, Any]:
    return dict(_row_metadata(row).get("example_metadata", {}))


def _declared_validation_available(row: BenchmarkRow) -> bool:
    example_metadata = _example_metadata(row)
    if "validation_supported" in example_metadata:
        return bool(example_metadata.get("validation_supported"))
    metadata = _row_metadata(row)
    if "validation_supported" in metadata:
        return bool(metadata.get("validation_supported"))
    return bool(row.semantic_validation_available)


def _executed_validation_available(row: BenchmarkRow) -> bool:
    return bool(row.semantic_validation_available)


def _unique_example_rows(rows: Iterable[BenchmarkRow]) -> list[BenchmarkRow]:
    unique: dict[str, BenchmarkRow] = {}
    for row in rows:
        unique.setdefault(row.example_id, row)
    return list(unique.values())


def _clean_functional_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _unique_example_rows(rows):
        trials = _row_metadata(row).get("clean_functional_trials", [])
        if isinstance(trials, list):
            grouped[row.example_id] = [dict(item) for item in trials if isinstance(item, Mapping)]

    if not grouped:
        return {"sample_count": 0.0, "test_pass_rate": 0.0, "pass@1": 0.0}

    all_trials = [trial for trials in grouped.values() for trial in trials]
    pass_values = [1.0 if trial.get("passed") else 0.0 for trial in all_trials if trial.get("passed") is not None]
    pass_at_1: list[float] = []
    for trials in grouped.values():
        successes = sum(1 for trial in trials if trial.get("passed") is True)
        pass_at_1.append(1.0 if successes > 0 else 0.0)
    return {
        "sample_count": float(len(all_trials)),
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
        "pass@1": round(mean(pass_at_1), 4) if pass_at_1 else 0.0,
    }


def _watermarked_functional_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    validations: list[dict[str, Any]] = []
    for row in _unique_example_rows(rows):
        clean_validation = _row_metadata(row).get("clean_validation", {})
        if isinstance(clean_validation, Mapping):
            validations.append(dict(clean_validation))
    available = [item for item in validations if item.get("available")]
    pass_values = [1.0 if item.get("passed") else 0.0 for item in available if item.get("passed") is not None]
    return {
        "validated_tasks": float(len(available)),
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
        "pass@1": round(mean(pass_values), 4) if pass_values else 0.0,
    }


def _attacked_functional_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, float]:
    validations: list[dict[str, Any]] = []
    for row in rows:
        attacked = _row_metadata(row).get("attacked_validation", {})
        if isinstance(attacked, Mapping):
            validations.append(dict(attacked))
    available = [item for item in validations if item.get("available")]
    pass_values = [1.0 if item.get("passed") else 0.0 for item in available if item.get("passed") is not None]
    return {
        "validated_rows": float(len(available)),
        "test_pass_rate": round(mean(pass_values), 4) if pass_values else 0.0,
    }


def _negative_control_applicable(row: BenchmarkRow, control: object, *, family: str) -> bool:
    if not isinstance(control, Mapping):
        return family == "human_reference"
    if "applicable" in control:
        return bool(control.get("applicable"))
    if family == "human_reference":
        return True
    if str(row.evaluation_track).strip() == "reference_code":
        return False
    metadata = _row_metadata(row)
    provider_mode = str(metadata.get("provider_mode", "")).strip().lower()
    if provider_mode in {"local_hf", "local_command", "watermark_runtime"}:
        return True
    example_metadata = _example_metadata(row)
    if "provider_generation_succeeded" in example_metadata:
        return True
    return bool(control.get("available"))


def _negative_control_metrics(rows: Iterable[BenchmarkRow]) -> dict[str, Any]:
    positives: list[float] = []
    negative_scores: list[float] = []
    negative_flags: list[bool] = []
    coverage = {
        "positive_examples": 0,
        "human_reference_negatives": 0,
        "clean_generation_negatives": 0,
        "human_reference_applicable": 0,
        "clean_generation_applicable": 0,
    }
    family_coverage_rates: dict[str, float] = {}

    for row in _unique_example_rows(rows):
        positives.append(float(row.clean_score))
        coverage["positive_examples"] += 1
        negative_controls = _row_metadata(row).get("negative_controls", {})
        if not isinstance(negative_controls, Mapping):
            negative_controls = {}
        for family in NEGATIVE_CONTROL_FAMILIES:
            control = negative_controls.get(family, {})
            applicable = _negative_control_applicable(row, control, family=family)
            if applicable:
                coverage[f"{family}_applicable"] += 1
            if not (applicable and isinstance(control, Mapping) and control.get("available")):
                continue
            negative_scores.append(float(control.get("score", 0.0)))
            negative_flags.append(bool(control.get("detected", False)))
            coverage[f"{family}_negatives"] += 1

    applicable_types: list[str] = []
    missing_types: list[str] = []
    observed_types: list[str] = []
    support_rates: list[float] = []
    for family in NEGATIVE_CONTROL_FAMILIES:
        applicable = int(coverage[f"{family}_applicable"])
        observed = int(coverage[f"{family}_negatives"])
        if applicable <= 0:
            continue
        applicable_types.append(family)
        rate = _clamp01(float(observed) / max(float(applicable), 1.0))
        family_coverage_rates[family] = round(rate, 4)
        support_rates.append(rate)
        if observed > 0:
            observed_types.append(family)
        else:
            missing_types.append(family)

    support_rate = round(mean(support_rates), 4) if support_rates else 0.0
    if negative_flags:
        fpr = mean(1.0 if detected else 0.0 for detected in negative_flags)
    elif applicable_types:
        fpr = 1.0
    else:
        fpr = 0.0
    auroc = binary_auroc(negative_scores, positives) if negative_scores and positives else 0.0

    coverage.update(
        {
            "negative_control_support_rate": support_rate,
            "negative_control_applicable_types": applicable_types,
            "negative_control_observed_types": observed_types,
            "negative_control_missing_types": missing_types,
            "negative_control_family_coverage": family_coverage_rates,
        }
    )
    return {
        "negative_control_fpr": round(_clamp01(fpr), 4),
        "negative_vs_watermarked_auroc": round(_clamp01(auroc), 4),
        "negative_control_support_rate": support_rate,
        "coverage": coverage,
    }


def _task_slice_key(row: BenchmarkRow) -> str:
    category = str(row.task_category or _example_metadata(row).get("category", "")).strip()
    if category:
        return category
    source_group = str(row.source_group or _example_metadata(row).get("source_group", "unspecified")).strip() or "unspecified"
    difficulty = str(row.difficulty or _example_metadata(row).get("difficulty", "unspecified")).strip() or "unspecified"
    reference_kind = str(row.reference_kind or _example_metadata(row).get("reference_kind", "unspecified")).strip() or "unspecified"
    return f"{source_group}:{difficulty}:{reference_kind}"


def _group_rows(rows: Iterable[BenchmarkRow], key_fn) -> dict[str, list[BenchmarkRow]]:
    grouped: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        key = str(key_fn(row)).strip() or "unspecified"
        grouped[key].append(row)
    return dict(grouped)


def _restrict_rows(rows: Iterable[BenchmarkRow], *, restrict_source_groups: Iterable[str] | None) -> list[BenchmarkRow]:
    materialized = list(rows)
    if restrict_source_groups is None:
        return materialized
    allowed = {normalize_source_group(value) for value in restrict_source_groups if normalize_source_group(value)}
    if not allowed:
        return []
    return [
        row
        for row in materialized
        if normalize_source_group(str(row.source_group or _example_metadata(row).get("source_group", ""))) in allowed
    ]


def _collapse_sparse_groups(groups: dict[str, list[BenchmarkRow]], *, min_rows: int) -> tuple[dict[str, list[BenchmarkRow]], int]:
    if not groups:
        return {}, 0
    dense: dict[str, list[BenchmarkRow]] = {}
    other: list[BenchmarkRow] = []
    folded = 0
    for name, group_rows in groups.items():
        if len(group_rows) < min_rows:
            other.extend(group_rows)
            folded += len(group_rows)
            continue
        dense[name] = group_rows
    if other:
        dense["other"] = other
    return dense or groups, folded


def _score_components_raw(rows: list[BenchmarkRow]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {
            "detection_reliability": 0.0,
            "robustness": 0.0,
            "utility": 0.0,
            "stealth": 0.0,
            "slice_core": 0.0,
            "negative_control_fpr": 1.0,
            "negative_vs_watermarked_auroc": 0.0,
            "negative_control_support_rate": 0.0,
            "semantic_validation_rate": 0.0,
            "declared_semantic_validation_rate": 0.0,
            "watermarked_pass_preservation": 0.0,
            "attacked_pass_preservation": 0.0,
            "score_coverage": {
                "positive_examples": 0,
                "human_reference_negatives": 0,
                "clean_generation_negatives": 0,
                "human_reference_applicable": 0,
                "clean_generation_applicable": 0,
                "negative_control_support_rate": 0.0,
                "negative_control_applicable_types": [],
                "negative_control_observed_types": [],
                "negative_control_missing_types": [],
                "negative_control_family_coverage": {},
            },
        }

    executed_rate = _clamp01(mean(1.0 if _executed_validation_available(row) else 0.0 for row in rows))
    declared_rate = _clamp01(mean(1.0 if _declared_validation_available(row) else 0.0 for row in rows))
    negative_metrics = _negative_control_metrics(rows)
    clean_functional = _clean_functional_metrics(rows)
    watermarked_functional = _watermarked_functional_metrics(rows)
    attacked_functional = _attacked_functional_metrics(rows)
    semantic_attack = mean(
        1.0 if row.attacked_detected else 0.0
        for row in rows
        if row.semantic_validation_available and row.semantic_preserving is True
    ) if any(row.semantic_validation_available and row.semantic_preserving is True for row in rows) else 0.0
    semantic_preservation = mean(
        1.0 if row.semantic_preserving else 0.0
        for row in rows
        if row.semantic_validation_available
    ) if any(row.semantic_validation_available for row in rows) else 0.0

    detection_reliability_raw = (
        0.6 * float(negative_metrics["negative_vs_watermarked_auroc"])
        + 0.4 * (1.0 - float(negative_metrics["negative_control_fpr"]))
    )
    detection_reliability = float(negative_metrics["negative_control_support_rate"]) * detection_reliability_raw
    robustness = (
        0.4 * (semantic_attack * executed_rate)
        + 0.35 * _clamp01(mean(row.watermark_retention for row in rows))
        + 0.25 * _clamp01(mean(1.0 if row.attacked_detected else 0.0 for row in rows))
    )
    watermarked_pass_preservation = _safe_ratio(
        float(watermarked_functional["pass@1"]),
        float(clean_functional["pass@1"]),
    )
    attacked_pass_preservation = _safe_ratio(
        float(attacked_functional["test_pass_rate"]),
        float(clean_functional["test_pass_rate"]),
    )
    utility = (
        0.35 * watermarked_pass_preservation
        + 0.25 * attacked_pass_preservation
        + 0.20 * _clamp01(mean(row.quality_score for row in rows))
        + 0.20 * (_clamp01(semantic_preservation) * executed_rate)
    )
    stealth = _clamp01(mean(row.stealth_score for row in rows))
    slice_core = 0.25 * detection_reliability + 0.35 * robustness + 0.30 * utility + 0.10 * stealth
    return {
        "detection_reliability": round(_clamp01(detection_reliability), 4),
        "robustness": round(_clamp01(robustness), 4),
        "utility": round(_clamp01(utility), 4),
        "stealth": round(stealth, 4),
        "slice_core": round(_clamp01(slice_core), 4),
        "negative_control_fpr": float(negative_metrics["negative_control_fpr"]),
        "negative_vs_watermarked_auroc": float(negative_metrics["negative_vs_watermarked_auroc"]),
        "negative_control_support_rate": float(negative_metrics["negative_control_support_rate"]),
        "semantic_validation_rate": round(executed_rate, 4),
        "declared_semantic_validation_rate": round(declared_rate, 4),
        "watermarked_pass_preservation": round(watermarked_pass_preservation, 4),
        "attacked_pass_preservation": round(attacked_pass_preservation, 4),
        "score_coverage": dict(negative_metrics["coverage"]),
    }


def _score_components(rows: list[BenchmarkRow], *, balance_by_source_group: bool = False) -> dict[str, Any]:
    components = _score_components_raw(rows)
    if not balance_by_source_group:
        return components

    source_groups = _group_rows(
        rows,
        lambda row: normalize_source_group(str(row.source_group or _example_metadata(row).get("source_group", "")))
        or "unspecified",
    )
    if len(source_groups) <= 1:
        coverage = dict(components.get("score_coverage", {}))
        coverage.update(
            {
                "aggregation_mode": "source_balanced",
                "source_group_count": len(source_groups),
                "aggregated_source_groups": sorted(source_groups),
            }
        )
        components["score_coverage"] = coverage
        return components

    per_source = {name: _score_components_raw(group_rows) for name, group_rows in source_groups.items()}
    for key in SOURCE_BALANCED_COMPONENT_FIELDS:
        values = [float(component.get(key, 0.0)) for component in per_source.values()]
        components[key] = round(_clamp01(mean(values)), 4) if values else 0.0

    coverage = dict(components.get("score_coverage", {}))
    coverage.update(
        {
            "aggregation_mode": "source_balanced",
            "source_group_count": len(source_groups),
            "aggregated_source_groups": sorted(source_groups),
            "source_balanced_sources": {
                source_group: {
                    "row_count": len(source_groups[source_group]),
                    "detection_reliability": float(component.get("detection_reliability", 0.0)),
                    "robustness": float(component.get("robustness", 0.0)),
                    "utility": float(component.get("utility", 0.0)),
                    "stealth": float(component.get("stealth", 0.0)),
                    "slice_core": float(component.get("slice_core", 0.0)),
                }
                for source_group, component in per_source.items()
            },
        }
    )
    components["score_coverage"] = coverage
    return components


def _stability_from_components(components: dict[str, dict[str, Any]]) -> float | None:
    if not components or len(components) < 2:
        return None
    values = [float(component.get("slice_core", 0.0)) for component in components.values()]
    if not values:
        return None
    baseline = mean(values)
    if baseline <= 0:
        return None
    return round(_clamp01(min(values) / max(baseline, EPSILON)), 4)


def _slice_component_map(
    groups: dict[str, list[BenchmarkRow]],
    *,
    balance_by_source_group: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        name: _score_components(group_rows, balance_by_source_group=balance_by_source_group)
        for name, group_rows in groups.items()
    }


def scorecard_for_rows(
    rows: Iterable[BenchmarkRow],
    *,
    include_generalization: bool = True,
    restrict_source_groups: Iterable[str] | None = None,
    balance_by_source_group: bool = False,
) -> dict[str, Any]:
    materialized = _restrict_rows(rows, restrict_source_groups=restrict_source_groups)
    components = _score_components(materialized, balance_by_source_group=balance_by_source_group)
    scorecard = dict(components)
    if not include_generalization:
        return scorecard

    model_groups = _group_rows(materialized, lambda row: row.model_label or "unspecified")
    source_groups = _group_rows(
        materialized,
        lambda row: normalize_source_group(str(row.source_group or _example_metadata(row).get("source_group", "")))
        or "unspecified",
    )
    task_groups, folded_rows = _collapse_sparse_groups(
        _group_rows(materialized, _task_slice_key),
        min_rows=SPARSE_TASK_MIN_ROWS,
    )
    model_components = _slice_component_map(model_groups, balance_by_source_group=balance_by_source_group)
    source_components = _slice_component_map(source_groups, balance_by_source_group=balance_by_source_group)
    task_components = _slice_component_map(task_groups, balance_by_source_group=balance_by_source_group)

    cross_model = _stability_from_components(model_components)
    cross_source = _stability_from_components(source_components)
    cross_task = _stability_from_components(task_components)
    available_axes = [value for value in (cross_model, cross_source, cross_task) if value is not None]
    generalization = round(_clamp01(mean(available_axes)), 4) if available_axes else 0.0
    base = _clamp01(
        0.20 * float(scorecard["detection_reliability"])
        + 0.25 * float(scorecard["robustness"])
        + 0.25 * float(scorecard["utility"])
        + 0.10 * float(scorecard["stealth"])
        + 0.20 * generalization
    )
    gate = _clamp01(
        min(
            1.0,
            float(scorecard["watermarked_pass_preservation"]),
            1.0 - float(scorecard["negative_control_fpr"]),
            float(scorecard["negative_control_support_rate"]),
        )
    )
    scorecard.update(
        {
            "generalization": generalization,
            "cross_model_stability": cross_model,
            "cross_source_stability": cross_source,
            "cross_task_stability": cross_task,
            "base_score": round(base, 4),
            "gate": round(gate, 4),
            "CodeWMScore": round(100.0 * base * gate, 4),
            "score_version": "codewmbench-suite-v1",
            "slice_core_by_model": model_components,
            "slice_core_by_source": source_components,
            "slice_core_by_task": task_components,
        }
    )
    coverage = dict(scorecard.get("score_coverage", {}))
    coverage.update(
        {
            "model_slice_count": len(model_components),
            "source_slice_count": len(source_components),
            "task_slice_count": len(task_components),
            "folded_sparse_task_rows": folded_rows,
            "generalization_axes_used": [
                axis_name
                for axis_name, axis_value in (
                    ("cross_model", cross_model),
                    ("cross_source", cross_source),
                    ("cross_task", cross_task),
                )
                if axis_value is not None
            ],
            "generalization_axes_missing": [
                axis_name
                for axis_name, axis_value in (
                    ("cross_model", cross_model),
                    ("cross_source", cross_source),
                    ("cross_task", cross_task),
                )
                if axis_value is None
            ],
        }
    )
    scorecard["score_coverage"] = coverage
    return scorecard
