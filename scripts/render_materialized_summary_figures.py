from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for candidate in (ROOT, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import render_paper_figures as rpf

DEFAULT_SUMMARY_DIR = ROOT / "results" / "figures" / "suite_all_models_methods"
DEFAULT_TABLE_DIR = ROOT / "results" / "tables" / "suite_all_models_methods"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw shipped full-run figures from materialized summary artifacts.")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--prefix", type=str, default="suite_all_models_methods")
    parser.add_argument(
        "--require-times-new-roman",
        dest="require_times_new_roman",
        action="store_true",
        default=True,
    )
    parser.add_argument("--allow-font-fallback", dest="require_times_new_roman", action="store_false")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _overall_method_order(rows: list[dict[str, Any]]) -> list[str]:
    sorted_rows = sorted(rows, key=lambda row: float(row.get("CodeWMScore", 0.0)), reverse=True)
    return [str(row.get("method", "")).strip() for row in sorted_rows if str(row.get("method", "")).strip()]


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return 0.0 if math.isnan(numeric) or math.isinf(numeric) else numeric


def _rows_by_key(rows: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key, "")).strip(): row for row in rows if str(row.get(key, "")).strip()}


def _render_overall(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    ordered = sorted(rows, key=lambda row: _safe_float(row.get("CodeWMScore", 0.0)), reverse=True)
    labels = [rpf._paper_label(str(row.get("method", ""))) for row in ordered]
    scores = [_safe_float(row.get("CodeWMScore", 0.0)) for row in ordered]
    axis_limit = rpf._score_axis_limit(scores)
    fig, ax = plt.subplots(
        figsize=(rpf.SINGLE_COLUMN_WIDTH, rpf._adaptive_track_figure_height(len(ordered), base=3.7, per_item=0.34, floor=3.7)),
        constrained_layout=True,
    )
    positions = list(range(len(ordered)))
    bars = ax.barh(
        positions,
        scores,
        color=[rpf._method_color(str(row.get("method", "")), str(row.get("origin", ""))) for row in ordered],
        edgecolor="white",
        linewidth=0.8,
    )
    for position, score, bar in zip(positions, scores, bars):
        if score > 0.0:
            ax.text(score + axis_limit * 0.025, position, f"{score:.1f}", va="center", ha="left", fontsize=9)
        else:
            ax.scatter([axis_limit * 0.01], [position], s=34, color=bar.get_facecolor(), edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(axis_limit * 0.035, position, "0.0", va="center", ha="left", fontsize=9)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, axis_limit)
    ax.set_xlabel("CodeWMScore")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _render_score_decomposition(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    ordered = sorted(rows, key=lambda row: _safe_float(row.get("base_score", 0.0)), reverse=True)
    metric_specs = [
        ("detection_reliability_contribution", "Det"),
        ("robustness_contribution", "Rob"),
        ("utility_contribution", "Utility"),
        ("stealth_contribution", "St"),
        ("generalization_contribution", "Gen"),
    ]
    fig, ax = plt.subplots(
        figsize=(rpf.SINGLE_COLUMN_WIDTH, rpf._adaptive_track_figure_height(len(ordered), base=3.65, per_item=0.3, floor=3.65)),
        constrained_layout=True,
    )
    positions = list(range(len(ordered)))
    labels = [rpf._paper_label(str(row.get("method", ""))) for row in ordered]
    cumulative = [0.0] * len(ordered)
    for key, label in metric_specs:
        values = [_safe_float(row.get(key, 0.0)) for row in ordered]
        ax.barh(
            positions,
            values,
            left=cumulative,
            color=rpf._METRIC_COLORS.get(
                {
                    "detection_reliability_contribution": "detection_reliability",
                    "robustness_contribution": "robustness",
                    "utility_contribution": "utility",
                    "stealth_contribution": "stealth",
                    "generalization_contribution": "generalization",
                }[key]
            ),
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        cumulative = [left + value for left, value in zip(cumulative, values)]
    for position, total in zip(positions, cumulative):
        ax.text(total + 0.012, position, f"{total:.2f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted Base Score")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0, columnspacing=0.9, handlelength=1.6)
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _render_generalization(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    ordered = sorted(rows, key=lambda row: _safe_float(row.get("generalization", 0.0)), reverse=True)
    fig, ax = plt.subplots(
        figsize=(rpf.SINGLE_COLUMN_WIDTH, rpf._adaptive_track_figure_height(len(ordered), base=3.45, per_item=0.28, floor=3.45)),
        constrained_layout=True,
    )
    labels = [rpf._paper_label(str(row.get("method", ""))) for row in ordered]
    values = [_safe_float(row.get("generalization", 0.0)) for row in ordered]
    positions = list(range(len(ordered)))
    ax.hlines(positions, [0.0] * len(values), values, color="#d7dde5", linewidth=2.2, zorder=1)
    for position, row, value in zip(positions, ordered, values):
        ax.scatter(value, position, s=70, color=rpf._method_color(str(row.get("method", "")), str(row.get("origin", ""))), edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(min(1.02, value + 0.02), position, f"{value:.2f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(0.0, 1.05)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Generalization")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _render_quality_vs_robustness(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(rpf.SINGLE_COLUMN_WIDTH, 3.55), constrained_layout=True)
    ax.axvline(0.5, color="#d7dde5", linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#d7dde5", linewidth=1.0, linestyle="--", zorder=1)
    ax.grid(alpha=0.25, linewidth=0.8)
    ordered = sorted(rows, key=lambda row: _safe_float(row.get("CodeWMScore", 0.0)), reverse=True)
    for index, row in enumerate(ordered):
        utility = _safe_float(row.get("utility", 0.0))
        robustness = _safe_float(row.get("robustness", 0.0))
        label = rpf._paper_label(str(row.get("method", "")))
        ax.scatter(
            utility,
            robustness,
            s=92,
            color=rpf._method_color(str(row.get("method", "")), str(row.get("origin", ""))),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset_x, offset_y = rpf._scatter_label_offset(index)
        ax.annotate(
            label,
            (utility, robustness),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Utility")
    ax.set_ylabel("Robustness")
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _render_functional_summary(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    return rpf._plot_functional_dotplot(plt, rows_payload=rows, title="", output_dir=output_dir, stem=stem)


def _render_source_breakdown(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    source_groups = [rpf.normalize_source_group(source_group) for source_group in rpf.SUITE_AGGREGATE_SOURCE_GROUPS]
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not methods:
        methods = sorted({str(row.get("method", "")).strip() for row in rows if str(row.get("method", "")).strip()})
    by_pair = {(str(row.get("method", "")).strip(), rpf.normalize_source_group(str(row.get("source_group", "")).strip())): row for row in rows}
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for source_group in source_groups:
            row = by_pair.get((method, source_group), {})
            value = _safe_float(row.get("slice_core", 0.0))
            values.append(value)
            payload.append(
                {
                    "method": method,
                    "paper_label": rpf._paper_label(method),
                    "source_group": source_group,
                    "source_label": rpf._suite_source_label(source_group),
                    "source_compact_label": rpf._paper_source_compact_label(source_group),
                    "slice_core": value,
                    "row_count": int(row.get("row_count", 0)) if row else 0,
                }
            )
        matrix.append(values)
    max_score = max((max(values) for values in matrix), default=0.0)
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[rpf._paper_source_compact_label(source_group) for source_group in source_groups],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="Blues",
        vmax=max(0.55, max_score + 0.04),
        colorbar_label="Slice Core",
        data=payload,
        max_columns_per_panel=len(source_groups),
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def _render_model_breakdown(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    model_order = [model for model in rpf.SUITE_MODEL_ROSTER if any(str(row.get("model", "")).strip() == model for row in rows)]
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not methods:
        methods = sorted({str(row.get("method", "")).strip() for row in rows if str(row.get("method", "")).strip()})
    by_pair = {(str(row.get("method", "")).strip(), str(row.get("model", "")).strip()): row for row in rows}
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for model in model_order:
            row = by_pair.get((method, model), {})
            value = _safe_float(row.get("CodeWMScore", 0.0))
            values.append(value)
            payload.append(
                {
                    "method": method,
                    "paper_label": rpf._paper_label(method),
                    "model": model,
                    "model_label": rpf._paper_model_label(model),
                    "CodeWMScore": value,
                    "row_count": int(row.get("row_count", 0)) if row else 0,
                }
            )
        matrix.append(values)
    max_score = max((max(values) for values in matrix), default=0.0)
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[rpf._paper_model_axis_label(model) for model in model_order],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="Blues",
        vmax=max(10.0, max_score + 1.0),
        data=payload,
        annotate=True,
        annotation_fmt="{:.1f}",
    )


def _render_language_breakdown(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    languages = [language for language in rpf.OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES if any(str(row.get("language", "")).strip() == language for row in rows)]
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not languages or not methods:
        return []
    by_pair = {(str(row.get("method", "")).strip(), str(row.get("language", "")).strip()): row for row in rows}
    fig_height = max(4.0, 1.55 * len(languages) + 0.65)
    fig, axes = plt.subplots(len(languages), 1, figsize=(rpf.SINGLE_COLUMN_WIDTH, fig_height), sharex=True, constrained_layout=True)
    if len(languages) == 1:
        axes = [axes]
    method_labels = [rpf._paper_label(method) for method in methods]
    for axis, language in zip(axes, languages):
        values = [_safe_float(by_pair.get((method, language), {}).get("utility", 0.0)) for method in methods]
        positions = list(range(len(methods)))
        axis.hlines(positions, [0.0] * len(values), values, color="#d7dde5", linewidth=2.2, zorder=1)
        for position, method, value in zip(positions, methods, values):
            axis.scatter(value, position, s=46, color=rpf._method_color(method), edgecolor="white", linewidth=0.8, zorder=3)
        axis.text(0.0, 1.01, language.title(), transform=axis.transAxes, ha="left", va="bottom", fontsize=10)
        axis.set_yticks(positions, method_labels)
        axis.set_xlim(0.0, 1.05)
        axis.grid(axis="x", alpha=0.25, linewidth=0.8)
        axis.invert_yaxis()
    axes[-1].set_xlabel("Utility")
    axes[len(axes) // 2].set_ylabel("Method")
    return rpf._save_figure(fig, output_dir, stem, data=rows)


def _render_attack_breakdown(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    attack_order = [
        "block_shuffle",
        "budgeted_adaptive",
        "comment_strip",
        "control_flow_flatten",
        "identifier_rename",
        "noise_insert",
        "whitespace_normalize",
    ]
    attacks = [attack for attack in attack_order if any(str(row.get("attack", "")).strip() == attack for row in rows)]
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not attacks or not methods:
        return []
    by_pair = {(str(row.get("method", "")).strip(), str(row.get("attack", "")).strip()): row for row in rows}
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        values: list[float] = []
        for attack in attacks:
            row = by_pair.get((method, attack), {})
            value = _safe_float(row.get("attacked_detect_rate", 0.0))
            values.append(value)
            payload.append(
                {
                    "method": method,
                    "paper_label": rpf._paper_label(method),
                    "attack": attack,
                    "attack_label": rpf._paper_attack_label(attack),
                    "attacked_detect_rate": value,
                    "row_count": int(row.get("row_count", 0)) if row else 0,
                }
            )
        matrix.append(values)
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[rpf._paper_attack_label(attack) for attack in attacks],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="Purples",
        data=payload,
        max_columns_per_panel=len(attacks),
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def _render_source_language_coverage(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    source_order = [source.source_group for source in rpf.SUITE_AGGREGATE_SOURCES]
    languages = ["python", "cpp", "java"]
    by_pair = {(rpf.normalize_source_group(str(row.get("source_group", "")).strip()), str(row.get("language", "")).strip().lower()): row for row in rows}
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for source_group in source_order:
        values: list[float] = []
        for language in languages:
            row = by_pair.get((rpf.normalize_source_group(source_group), language), {})
            value = _safe_float(row.get("count", 0))
            values.append(value)
            payload.append(
                {
                    "source_group": rpf.normalize_source_group(source_group),
                    "source_label": rpf._suite_source_label(rpf.normalize_source_group(source_group)),
                    "language": language,
                    "count": int(value),
                }
            )
        matrix.append(values)
    max_value = max((max(values) for values in matrix), default=0.0)
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_source_compact_label(source_group) for source_group in source_order],
        col_labels=[language.title() for language in languages],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(1.0, max_value),
        colorbar_label="Compact Records",
        data=payload,
        annotate=True,
        annotation_fmt="{:.0f}",
    )


def _render_method_stability(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str, method_order: list[str]) -> list[Path]:
    methods = [method for method in method_order if any(str(row.get("method", "")).strip() == method for row in rows)]
    if not methods:
        methods = sorted({str(row.get("method", "")).strip() for row in rows if str(row.get("method", "")).strip()})
    by_method = _rows_by_key(rows, key="method")
    metric_names = [
        ("cross_model_stability", "Model"),
        ("cross_source_stability", "Source"),
        ("cross_task_stability", "Task"),
    ]
    matrix: list[list[float]] = []
    payload: list[dict[str, Any]] = []
    for method in methods:
        row = by_method.get(method, {})
        values = [_safe_float(row.get(metric_name, 0.0)) for metric_name, _ in metric_names]
        matrix.append(values)
        payload.append({"method": method, "paper_label": rpf._paper_label(method), **{metric_name: _safe_float(row.get(metric_name, 0.0)) for metric_name, _ in metric_names}})
    return rpf._plot_heatmap(
        plt,
        matrix=matrix,
        row_labels=[rpf._paper_label(method) for method in methods],
        col_labels=[label for _, label in metric_names],
        title="",
        xlabel="",
        ylabel="",
        output_dir=output_dir,
        stem=stem,
        cmap="Blues",
        data=payload,
        annotate=True,
        annotation_fmt="{:.2f}",
    )


def _render_detection_vs_utility(plt, rows: list[dict[str, Any]], *, output_dir: Path, stem: str) -> list[Path]:
    ordered = sorted(rows, key=lambda row: _safe_float(row.get("CodeWMScore", 0.0)), reverse=True)
    fig, ax = plt.subplots(figsize=(rpf.SINGLE_COLUMN_WIDTH, 3.45), constrained_layout=True)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.axvline(0.5, color="#d7dde5", linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#d7dde5", linewidth=1.0, linestyle="--", zorder=1)
    ax.grid(alpha=0.25, linewidth=0.8)
    for index, row in enumerate(ordered):
        detection = _safe_float(row.get("detection_reliability", 0.0))
        utility = _safe_float(row.get("utility", 0.0))
        label = rpf._paper_label(str(row.get("method", "")))
        ax.scatter(
            detection,
            utility,
            s=92,
            color=rpf._method_color(str(row.get("method", "")), str(row.get("origin", ""))),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset_x, offset_y = rpf._scatter_label_offset(index)
        ax.annotate(
            label,
            (detection, utility),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    ax.set_xlabel("Detection Reliability")
    ax.set_ylabel("Utility")
    return rpf._save_figure(fig, output_dir, stem, data=ordered)


def _remove_duplicate_leaderboard_sidecars(output_dir: Path, prefix: str) -> None:
    for stem in (
        f"{prefix}_overall_leaderboard",
        f"{prefix}_public_only_overall_leaderboard",
    ):
        for suffix in (".json", ".csv"):
            path = output_dir / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.resolve()
    table_dir = args.table_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, plt = rpf.configure_matplotlib(require_times_new_roman=args.require_times_new_roman)

    overall_rows = _load_rows(table_dir / f"{args.prefix}_method_master_leaderboard.json")
    public_only_rows = _load_rows(table_dir / f"{args.prefix}_public_only_method_master_leaderboard.json")
    score_rows = _load_rows(summary_dir / f"{args.prefix}_score_decomposition.json")
    generalization_rows = _load_rows(summary_dir / f"{args.prefix}_generalization_breakdown.json")
    quality_rows = _load_rows(summary_dir / f"{args.prefix}_quality_vs_robustness.json")
    functional_rows = _load_rows(summary_dir / f"{args.prefix}_functional_summary.json")
    per_language_rows = _load_rows(summary_dir / f"{args.prefix}_per_language_breakdown.json")
    source_language_rows = _load_rows(summary_dir / f"{args.prefix}_source_language_coverage.json")
    stability_rows = _load_rows(summary_dir / f"{args.prefix}_method_stability_heatmap.json")
    detection_rows = _load_rows(summary_dir / f"{args.prefix}_detection_vs_utility.json")
    method_source_rows = _load_rows(table_dir / "method_source_summary.json")
    model_method_rows = _load_rows(table_dir / "model_method_summary.json")
    method_attack_rows = _load_rows(table_dir / "method_attack_summary.json")

    method_order = _overall_method_order(overall_rows)

    _render_overall(plt, overall_rows, output_dir=output_dir, stem=f"{args.prefix}_overall_leaderboard")
    _render_overall(plt, public_only_rows, output_dir=output_dir, stem=f"{args.prefix}_public_only_overall_leaderboard")
    _render_score_decomposition(plt, score_rows, output_dir=output_dir, stem=f"{args.prefix}_score_decomposition")
    _render_generalization(plt, generalization_rows, output_dir=output_dir, stem=f"{args.prefix}_generalization_breakdown")
    _render_quality_vs_robustness(plt, quality_rows, output_dir=output_dir, stem=f"{args.prefix}_quality_vs_robustness")
    _render_functional_summary(plt, functional_rows, output_dir=output_dir, stem=f"{args.prefix}_functional_summary")
    _render_source_breakdown(plt, method_source_rows, output_dir=output_dir, stem=f"{args.prefix}_per_source_breakdown", method_order=method_order)
    _render_model_breakdown(plt, model_method_rows, output_dir=output_dir, stem=f"{args.prefix}_per_model_breakdown", method_order=method_order)
    _render_language_breakdown(plt, per_language_rows, output_dir=output_dir, stem=f"{args.prefix}_per_language_breakdown", method_order=method_order)
    _render_attack_breakdown(plt, method_attack_rows, output_dir=output_dir, stem=f"{args.prefix}_attack_breakdown", method_order=method_order)
    _render_source_language_coverage(plt, source_language_rows, output_dir=output_dir, stem=f"{args.prefix}_source_language_coverage")
    _render_method_stability(plt, stability_rows, output_dir=output_dir, stem=f"{args.prefix}_method_stability_heatmap", method_order=method_order)
    _render_detection_vs_utility(plt, detection_rows, output_dir=output_dir, stem=f"{args.prefix}_detection_vs_utility")
    _remove_duplicate_leaderboard_sidecars(output_dir, args.prefix)


if __name__ == "__main__":
    main()
