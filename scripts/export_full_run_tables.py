from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codewmbench.models import BenchmarkRow
from codewmbench.report import (
    _attacked_functional_metrics,
    _clean_functional_metrics,
    _watermarked_functional_metrics,
)
from codewmbench.scorecard import scorecard_for_rows
from codewmbench.suite import normalize_source_group


_BENCHMARK_ROW_FIELDS = {field.name for field in fields(BenchmarkRow)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-run aggregate tables from existing report.json files.")
    parser.add_argument("--matrix-index", type=Path, required=True, help="Finished matrix_index.json path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON summary tables.")
    return parser.parse_args()


def _repo_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def _row_from_payload(payload: Mapping[str, Any]) -> BenchmarkRow:
    data = {key: value for key, value in payload.items() if key in _BENCHMARK_ROW_FIELDS}
    data.setdefault("metadata", {})
    return BenchmarkRow(**data)


def _round(value: float) -> float:
    return round(float(value), 4)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_rows_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_table(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> None:
    _write_rows_json(output_dir / f"{stem}.json", rows)
    _write_rows_csv(output_dir / f"{stem}.csv", rows)


def _run_duration_seconds(run: Mapping[str, Any]) -> float:
    return float(run.get("duration_seconds", 0.0) or 0.0)


def _group_rows(
    grouped_rows: Mapping[Any, list[BenchmarkRow]],
    grouped_runs: Mapping[Any, list[Mapping[str, Any]]],
    *,
    extra_fields: callable | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, benchmark_rows in grouped_rows.items():
        run_records = list(grouped_runs.get(key, []))
        scorecard = scorecard_for_rows(benchmark_rows, balance_by_source_group=True)
        clean = _clean_functional_metrics(benchmark_rows)
        watermarked = _watermarked_functional_metrics(benchmark_rows)
        attacked = _attacked_functional_metrics(benchmark_rows)
        durations = [_run_duration_seconds(run) for run in run_records]
        payload = {
            "row_count": len(benchmark_rows),
            "report_count": len(run_records),
            "duration_seconds_total": _round(sum(durations)),
            "duration_hours_total": _round(sum(durations) / 3600.0),
            "duration_seconds_mean": _round(mean(durations)) if durations else 0.0,
            "duration_hours_mean": _round(mean(durations) / 3600.0) if durations else 0.0,
            "CodeWMScore": float(scorecard.get("CodeWMScore", 0.0)),
            "detection_reliability": float(scorecard.get("detection_reliability", 0.0)),
            "robustness": float(scorecard.get("robustness", 0.0)),
            "utility": float(scorecard.get("utility", 0.0)),
            "stealth": float(scorecard.get("stealth", 0.0)),
            "generalization": float(scorecard.get("generalization", 0.0)),
            "slice_core": float(scorecard.get("slice_core", 0.0)),
            "negative_control_fpr": float(scorecard.get("negative_control_fpr", 0.0)),
            "negative_control_support_rate": float(scorecard.get("negative_control_support_rate", 0.0)),
            "clean_compile_success_rate": float(clean.get("compile_success_rate", 0.0)),
            "clean_test_pass_rate": float(clean.get("test_pass_rate", 0.0)),
            "clean_pass@1": float(clean.get("pass@1", 0.0)),
            "watermarked_test_pass_rate": float(watermarked.get("test_pass_rate", 0.0)),
            "watermarked_pass@1": float(watermarked.get("pass@1", 0.0)),
            "attacked_test_pass_rate": float(attacked.get("test_pass_rate", 0.0)),
            "watermarked_pass_preservation": float(scorecard.get("watermarked_pass_preservation", 0.0)),
            "attacked_pass_preservation": float(scorecard.get("attacked_pass_preservation", 0.0)),
        }
        if extra_fields is not None:
            payload.update(extra_fields(key, benchmark_rows, run_records))
        rows.append(payload)
    return rows


def main() -> int:
    args = _parse_args()
    matrix_index_path = args.matrix_index.resolve()
    base_dir = matrix_index_path.parents[3]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_index = json.loads(matrix_index_path.read_text(encoding="utf-8"))
    runs = [run for run in matrix_index.get("runs", []) if isinstance(run, Mapping) and str(run.get("status")) == "success"]

    reports: list[tuple[Mapping[str, Any], list[BenchmarkRow], Mapping[str, Any]]] = []
    for run in runs:
        report_path = _repo_path(run["report_path"], base_dir=base_dir)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        rows = [_row_from_payload(row) for row in payload.get("rows", []) if isinstance(row, Mapping)]
        reports.append((run, rows, payload))

    by_method_rows: dict[str, list[BenchmarkRow]] = defaultdict(list)
    by_model_rows: dict[str, list[BenchmarkRow]] = defaultdict(list)
    by_model_method_rows: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    by_method_source_rows: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    by_method_attack_rows: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)

    by_method_runs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_model_runs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_model_method_runs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_method_source_runs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_method_attack_runs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)

    run_inventory: list[dict[str, Any]] = []

    for run, rows, _payload in reports:
        if not rows:
            continue
        exemplar = rows[0]
        method = str(exemplar.watermark_scheme).strip() or "unspecified"
        model = str(exemplar.model_label).strip() or "unspecified"
        source = normalize_source_group(str(exemplar.source_group).strip() or str(exemplar.dataset).strip()) or "unspecified"

        by_method_rows[method].extend(rows)
        by_model_rows[model].extend(rows)
        by_model_method_rows[(model, method)].extend(rows)
        by_method_source_rows[(method, source)].extend(rows)
        by_method_runs[method].append(run)
        by_model_runs[model].append(run)
        by_model_method_runs[(model, method)].append(run)
        by_method_source_runs[(method, source)].append(run)

        run_inventory.append(
            {
                "run_id": str(run.get("run_id", "")),
                "status": str(run.get("status", "")),
                "duration_hours": _round(_run_duration_seconds(run) / 3600.0),
                "method": method,
                "model": model,
                "source_group": source,
                "row_count": len(rows),
                "CodeWMScore": float(scorecard_for_rows(rows, balance_by_source_group=True).get("CodeWMScore", 0.0)),
            }
        )

        attack_groups: dict[str, list[BenchmarkRow]] = defaultdict(list)
        for row in rows:
            attack = str(row.attack_name).strip() or "unspecified"
            attack_groups[attack].append(row)
        for attack, attack_rows in attack_groups.items():
            by_method_attack_rows[(method, attack)].extend(attack_rows)
            by_method_attack_runs[(method, attack)].append(run)

    method_summary = _group_rows(by_method_rows, by_method_runs, extra_fields=lambda key, _rows, _runs: {"method": key})
    method_summary.sort(key=lambda row: (-float(row["CodeWMScore"]), str(row["method"])))

    model_summary = _group_rows(by_model_rows, by_model_runs, extra_fields=lambda key, _rows, _runs: {"model": key})
    model_summary.sort(key=lambda row: (-float(row["CodeWMScore"]), str(row["model"])))

    model_method_summary = _group_rows(
        by_model_method_rows,
        by_model_method_runs,
        extra_fields=lambda key, _rows, _runs: {"model": key[0], "method": key[1]},
    )
    model_method_summary.sort(key=lambda row: (str(row["model"]), -float(row["CodeWMScore"]), str(row["method"])))

    method_source_summary = _group_rows(
        by_method_source_rows,
        by_method_source_runs,
        extra_fields=lambda key, _rows, _runs: {"method": key[0], "source_group": key[1]},
    )
    method_source_summary.sort(key=lambda row: (str(row["source_group"]), -float(row["CodeWMScore"]), str(row["method"])))

    method_attack_summary = _group_rows(
        by_method_attack_rows,
        by_method_attack_runs,
        extra_fields=lambda key, rows, _runs: {
            "method": key[0],
            "attack": key[1],
            "attacked_detect_rate": _round(sum(1.0 for row in rows if bool(row.attacked_detected)) / max(len(rows), 1)),
            "mean_attacked_score": _round(mean(row.attacked_score for row in rows) if rows else 0.0),
            "mean_quality_score": _round(mean(row.quality_score for row in rows) if rows else 0.0),
        },
    )
    method_attack_summary.sort(key=lambda row: (str(row["attack"]), -float(row["attacked_detect_rate"]), str(row["method"])))

    timing_summary = [
        {
            "method": row["method"],
            "duration_hours_total": row["duration_hours_total"],
            "duration_hours_mean": row["duration_hours_mean"],
            "report_count": row["report_count"],
        }
        for row in method_summary
    ]
    timing_summary.sort(key=lambda row: (-float(row["duration_hours_total"]), str(row["method"])))

    run_inventory.sort(key=lambda row: (str(row["model"]), str(row["method"]), str(row["source_group"])))

    _write_table(output_dir, "method_summary", method_summary)
    _write_table(output_dir, "model_summary", model_summary)
    _write_table(output_dir, "model_method_summary", model_method_summary)
    _write_table(output_dir, "method_source_summary", method_source_summary)
    _write_table(output_dir, "method_attack_summary", method_attack_summary)
    _write_table(output_dir, "timing_summary", timing_summary)
    _write_table(output_dir, "suite_all_models_methods_run_inventory", run_inventory)
    print(f"wrote tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
