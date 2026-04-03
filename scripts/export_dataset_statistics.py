from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

try:
    from _shared import dump_json, read_jsonl
except ModuleNotFoundError:  # pragma: no cover
    from scripts._shared import dump_json, read_jsonl

from codewmbench.suite import OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES
from scripts.render_paper_figures import SINGLE_COLUMN_WIDTH, configure_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dataset statistics tables and single-column figures for the public repo.")
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=ROOT / "results" / "tables" / "dataset_statistics",
        help="Output directory for dataset statistics tables.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "results" / "figures" / "dataset_statistics",
        help="Output directory for dataset statistics figures.",
    )
    parser.add_argument(
        "--require-times-new-roman",
        dest="require_times_new_roman",
        action="store_true",
        default=True,
        help="Fail closed if Times New Roman is unavailable.",
    )
    parser.add_argument(
        "--allow-font-fallback",
        dest="require_times_new_roman",
        action="store_false",
        help="Allow serif fallback fonts instead of failing on missing Times New Roman.",
    )
    return parser.parse_args()


INVENTORY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "slug": "human_eval",
        "dataset_label": "HumanEval",
        "source_group": "public_human_eval",
        "scope": "inventory",
        "path": ROOT / "data" / "public" / "human_eval" / "normalized.jsonl",
        "manifest_path": ROOT / "data" / "public" / "human_eval" / "normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": False,
    },
    {
        "slug": "humaneval_plus",
        "dataset_label": "HumanEval+",
        "source_group": "public_humaneval_plus",
        "scope": "inventory",
        "path": ROOT / "data" / "public" / "humaneval_plus" / "normalized.jsonl",
        "manifest_path": ROOT / "data" / "public" / "humaneval_plus" / "normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
    },
    {
        "slug": "mbpp_plus",
        "dataset_label": "MBPP+",
        "source_group": "public_mbpp_plus",
        "scope": "inventory",
        "path": ROOT / "data" / "public" / "mbpp_plus" / "normalized.jsonl",
        "manifest_path": ROOT / "data" / "public" / "mbpp_plus" / "normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
    },
    {
        "slug": "humaneval_x",
        "dataset_label": "HumanEval-X",
        "source_group": "public_humaneval_x",
        "scope": "inventory",
        "path": ROOT / "data" / "public" / "humaneval_x" / "normalized.jsonl",
        "manifest_path": ROOT / "data" / "public" / "humaneval_x" / "normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
    },
    {
        "slug": "mbxp_5lang",
        "dataset_label": "MBXP-5lang",
        "source_group": "public_mbxp_5lang",
        "scope": "inventory",
        "path": ROOT / "data" / "public" / "mbxp_5lang" / "normalized.jsonl",
        "manifest_path": ROOT / "data" / "public" / "mbxp_5lang" / "normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
    },
    {
        "slug": "crafted_original",
        "dataset_label": "Crafted Original",
        "source_group": "crafted_original",
        "scope": "inventory",
        "path": ROOT / "data" / "compact" / "crafted" / "crafted_original.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "crafted" / "crafted_original.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
    },
    {
        "slug": "crafted_translation",
        "dataset_label": "Crafted Translation",
        "source_group": "crafted_translation",
        "scope": "inventory",
        "path": ROOT / "data" / "compact" / "crafted" / "crafted_translation.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "crafted" / "crafted_translation.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
    },
    {
        "slug": "crafted_stress",
        "dataset_label": "Crafted Stress",
        "source_group": "crafted_stress",
        "scope": "inventory",
        "path": ROOT / "data" / "compact" / "crafted" / "crafted_stress.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "crafted" / "crafted_stress.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
    },
)

COMPACT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "slug": "humaneval_plus",
        "dataset_label": "HumanEval+",
        "source_group": "public_humaneval_plus",
        "scope": "compact",
        "path": ROOT / "data" / "public" / "humaneval_plus" / "normalized.jsonl",
        "manifest_path": ROOT / "data" / "public" / "humaneval_plus" / "normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
        "execution_slice": "python",
    },
    {
        "slug": "mbpp_plus",
        "dataset_label": "MBPP+",
        "source_group": "public_mbpp_plus",
        "scope": "compact",
        "path": ROOT / "data" / "compact" / "collections" / "suite_mbpp_plus_compact.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "collections" / "suite_mbpp_plus_compact.normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
        "execution_slice": "python",
    },
    {
        "slug": "humaneval_x",
        "dataset_label": "HumanEval-X (py/cpp/java slice)",
        "source_group": "public_humaneval_x",
        "scope": "compact",
        "path": ROOT / "data" / "compact" / "collections" / "suite_humanevalx_compact.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "collections" / "suite_humanevalx_compact.normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
    {
        "slug": "mbxp_5lang",
        "dataset_label": "MBXP-5lang (py/cpp/java slice)",
        "source_group": "public_mbxp_5lang",
        "scope": "compact",
        "path": ROOT / "data" / "compact" / "collections" / "suite_mbxp_compact.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "collections" / "suite_mbxp_compact.normalized.manifest.json",
        "source_type": "public",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
    {
        "slug": "crafted_original",
        "dataset_label": "Crafted Original",
        "source_group": "crafted_original",
        "scope": "compact",
        "path": ROOT / "data" / "compact" / "collections" / "suite_crafted_original_compact.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "collections" / "suite_crafted_original_compact.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
    {
        "slug": "crafted_translation",
        "dataset_label": "Crafted Translation",
        "source_group": "crafted_translation",
        "scope": "compact",
        "path": ROOT / "data" / "compact" / "collections" / "suite_crafted_translation_compact.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "collections" / "suite_crafted_translation_compact.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
    {
        "slug": "crafted_stress",
        "dataset_label": "Crafted Stress",
        "source_group": "crafted_stress",
        "scope": "compact",
        "path": ROOT / "data" / "compact" / "collections" / "suite_crafted_stress_compact.normalized.jsonl",
        "manifest_path": ROOT / "data" / "compact" / "collections" / "suite_crafted_stress_compact.normalized.manifest.json",
        "source_type": "crafted",
        "aggregate_score": True,
        "execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
    },
)

_PUBLIC_PANEL_COLORS = ["#355C7D", "#427AA1", "#5C94BF", "#7FB3D5", "#A7C8E6"]
_CRAFTED_PANEL_COLORS = ["#C06C84", "#D17C48", "#8C6F56"]
_SOURCE_TYPE_COLORS = {"public": "#355C7D", "crafted": "#D17C48"}
_LANGUAGE_HEATMAP = "YlGnBu"
_CATEGORY_HEATMAP = "YlOrRd"
_FAMILY_HEATMAP = "PuBuGn"

_SOURCE_SHORT_LABELS = {
    "HumanEval": "HE",
    "HumanEval+": "HE+",
    "MBPP+": "MBPP+",
    "HumanEval-X": "HEX",
    "HumanEval-X (py/cpp/java slice)": "HEX",
    "MBXP-5lang": "MBXP",
    "MBXP-5lang (py/cpp/java slice)": "MBXP",
    "Crafted Original": "Orig.",
    "Crafted Translation": "Trans.",
    "Crafted Stress": "Stress",
}

_COMPACT_FIGURE_SOURCE_LABELS = {
    "HumanEval+": "HE+\n(active)",
    "MBPP+": "MBPP+\n(active)",
    "HumanEval-X (py/cpp/java slice)": "HEX\n(py/cpp/java)",
    "MBXP-5lang (py/cpp/java slice)": "MBXP-5lang\n(py/cpp/java)",
    "Crafted Original": "Orig.\n(active)",
    "Crafted Translation": "Trans.\n(active)",
    "Crafted Stress": "Stress\n(active)",
}

_INVENTORY_FIGURE_SOURCE_LABELS = {
    "HumanEval": "HumanEval",
    "HumanEval+": "HE+",
    "MBPP+": "MBPP+",
    "HumanEval-X": "HEX\n(5-lang inv.)",
    "MBXP-5lang": "MBXP-5lang\n(5-lang inv.)",
    "Crafted Original": "Orig.\n(inv.)",
    "Crafted Translation": "Trans.\n(inv.)",
    "Crafted Stress": "Stress\n(inv.)",
}

_CATEGORY_SHORT_LABELS = {
    "class/object interaction": "Object\ninteraction",
    "cross-language idiom preservation": "Cross-lang\nidioms",
    "data structures": "Data\nstructures",
    "exception/error handling": "Errors",
    "graph/search": "Graph /\nsearch",
    "numeric/boundary conditions": "Numeric /\nboundary",
    "recursion/dp": "Recursion /\nDP",
    "state machines/simulation": "State /\nsim.",
    "strings/parsing": "Strings /\nparsing",
    "Other categories": "Other\ncats.",
}

_FAMILY_SHORT_LABELS = {
    "API-style normalization": "API\nnormalize",
    "arrays/lists": "Arrays /\nlists",
    "dp/recursion": "DP /\nrec.",
    "graph/search": "Graph /\nsearch",
    "maps/sets": "Maps /\nsets",
    "math/bit ops": "Math /\nbit",
    "parsing": "Parse",
    "stateful update": "Stateful\nupdate",
    "strings": "Strings",
    "Other families": "Other\nfam.",
}

_ACTIVE_RELEASE_POLICY: dict[str, dict[str, Any]] = {
    "human_eval": {
        "active_execution_slice": "",
        "sampling_rule": "inventory only; excluded from aggregate scoring",
        "inventory_scored": False,
        "compact_scored": False,
    },
    "humaneval_plus": {
        "active_execution_slice": "python",
        "sampling_rule": "full retained",
        "inventory_scored": True,
        "compact_scored": True,
    },
    "mbpp_plus": {
        "active_execution_slice": "python",
        "sampling_rule": "deterministic compact sample",
        "inventory_scored": False,
        "compact_scored": True,
    },
    "humaneval_x": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "deterministic common-support slice",
        "inventory_scored": False,
        "compact_scored": True,
    },
    "mbxp_5lang": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "deterministic common-support slice",
        "inventory_scored": False,
        "compact_scored": True,
    },
    "crafted_original": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "deterministic family/category-balanced compact sample",
        "inventory_scored": False,
        "compact_scored": True,
    },
    "crafted_translation": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "deterministic family/category-balanced compact sample",
        "inventory_scored": False,
        "compact_scored": True,
    },
    "crafted_stress": {
        "active_execution_slice": "/".join(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES),
        "sampling_rule": "deterministic family/category-balanced compact sample",
        "inventory_scored": False,
        "compact_scored": True,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _label_map(rows: Iterable[dict[str, Any]], field: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        value = _normalize(row.get(field))
        raw = str(row.get(field, "")).strip()
        if value and raw and value not in labels:
            labels[value] = raw
    return labels


def _count_field(rows: list[dict[str, Any]], field: str) -> tuple[dict[str, int], dict[str, str]]:
    labels = _label_map(rows, field)
    counts = Counter(_normalize(row.get(field)) for row in rows if _normalize(row.get(field)))
    return dict(sorted(counts.items())), labels


def _spec_summary(spec: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(spec["path"])
    manifest = _load_json(spec["manifest_path"])
    release_policy = _ACTIVE_RELEASE_POLICY[spec["slug"]]
    language_counts, language_labels = _count_field(rows, "language")
    category_counts, category_labels = _count_field(rows, "category")
    family_ids = {str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip()}
    template_family_counts, template_family_labels = _count_field(rows, "template_family")
    if spec["scope"] == "inventory":
        aggregate_score = bool(release_policy["inventory_scored"] or release_policy["compact_scored"])
        execution_slice = ""
        active_execution_slice = ""
        sampling_rule = "inventory snapshot"
        scoring_status = "inventory_documented"
    else:
        aggregate_score = bool(release_policy["compact_scored"])
        execution_slice = str(spec.get("execution_slice", release_policy["active_execution_slice"]))
        active_execution_slice = str(release_policy["active_execution_slice"])
        sampling_rule = str(release_policy["sampling_rule"])
        scoring_status = "active_compact_slice_scored" if aggregate_score else "active_compact_slice_not_scored"
    return {
        "slug": spec["slug"],
        "dataset_label": spec["dataset_label"],
        "source_group": spec["source_group"],
        "scope": spec["scope"],
        "source_type": spec["source_type"],
        "aggregate_score": aggregate_score,
        "execution_slice": execution_slice,
        "active_execution_slice": active_execution_slice,
        "sampling_rule": sampling_rule,
        "scoring_status": scoring_status,
        "path": str(Path(spec["path"]).relative_to(ROOT)).replace("\\", "/"),
        "manifest_path": str(Path(spec["manifest_path"]).relative_to(ROOT)).replace("\\", "/"),
        "record_count": int(len(rows)),
        "language_count": int(len(language_counts)),
        "languages": [language_labels.get(key, key) for key in language_counts],
        "language_counts": language_counts,
        "category_count": int(len(category_counts)),
        "category_counts": category_counts,
        "category_labels": category_labels,
        "family_count": int(len(family_ids) or manifest.get("family_count") or 0),
        "template_family_count": int(len(template_family_counts)),
        "template_family_counts": template_family_counts,
        "template_family_labels": template_family_labels,
        "validation_scope": str(manifest.get("validation_scope", "")),
        "source_manifest_count": int(len(manifest.get("source_manifests", []) or [])),
        "claimed_languages": list(manifest.get("claimed_languages", manifest.get("languages", [])) or []),
    }


def _write_json_csv(path_root: Path, rows: list[dict[str, Any]]) -> None:
    path_root.parent.mkdir(parents=True, exist_ok=True)
    dump_json(path_root.with_suffix(".json"), rows)
    if not rows:
        path_root.with_suffix(".csv").write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path_root.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _source_short_label(value: str) -> str:
    return _SOURCE_SHORT_LABELS.get(str(value).strip(), str(value).strip())


def _compact_figure_source_label(value: str) -> str:
    raw = str(value).strip()
    return _COMPACT_FIGURE_SOURCE_LABELS.get(raw, _source_short_label(raw))


def _inventory_figure_source_label(value: str) -> str:
    raw = str(value).strip()
    return _INVENTORY_FIGURE_SOURCE_LABELS.get(raw, _source_short_label(raw))


def _category_short_label(value: str) -> str:
    return _CATEGORY_SHORT_LABELS.get(str(value).strip(), str(value).strip().title())


def _family_short_label(value: str) -> str:
    return _FAMILY_SHORT_LABELS.get(str(value).strip(), str(value).strip())


def _save_plot(fig: Any, output_root: Path) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_root.with_suffix(".png"), bbox_inches="tight")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _annotate_hbar_end(ax: Any, values: list[int | float], y_positions: list[int], *, pad_fraction: float = 0.015) -> None:
    if not values:
        return
    max_value = max(float(value) for value in values)
    pad = max(1.0, max_value * pad_fraction)
    for y_position, value in zip(y_positions, values):
        ax.text(
            float(value) + pad,
            y_position,
            f"{int(value)}",
            va="center",
            ha="left",
            fontsize=9,
            color="#243B53",
        )


def _draw_count_heatmap(
    plt: Any,
    *,
    matrix: list[list[int]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_root: Path,
    cmap: str,
    colorbar_label: str,
    subtitle: str = "",
) -> None:
    fig_height = max(2.45, 1.6 + 0.27 * len(row_labels))
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, fig_height), constrained_layout=True)
    vmax = max(max(row) for row in matrix) if matrix else 1
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, vmax))
    if title:
        fig.suptitle(title, x=0.02, ha="left", fontsize=12.8)
    if subtitle:
        fig.text(0.02, 0.94, subtitle, ha="left", va="top", fontsize=9, color="#52606D")
    ax.set_xticks(list(range(len(col_labels))), col_labels)
    ax.set_yticks(list(range(len(row_labels))), row_labels)
    ax.tick_params(axis="x", rotation=0, labelsize=8.8)
    ax.tick_params(axis="y", labelsize=9.4)
    threshold = max(1, vmax) * 0.58
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            text_color = "white" if value >= threshold and vmax > 0 else "#102A43"
            ax.text(column_index, row_index, str(int(value)), ha="center", va="center", fontsize=8.4, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=12)
    _save_plot(fig, output_root)
    plt.close(fig)


def _draw_flow_box(ax: Any, x: float, y: float, width: float, height: float, title: str, subtitle: str, facecolor: str) -> tuple[float, float, float, float]:
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.1,
        edgecolor="#2F4858",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2.0, y + height * 0.62, title, ha="center", va="center", fontsize=10, color="#102A43")
    if subtitle:
        ax.text(x + width / 2.0, y + height * 0.28, subtitle, ha="center", va="center", fontsize=8.6, color="#52606D")
    return (x, y, width, height)


def _inventory_source_counts_figure(plt: Any, inventory_rows: list[dict[str, Any]], output_root: Path) -> None:
    public_rows = [row for row in inventory_rows if row["source_type"] == "public"]
    crafted_rows = [row for row in inventory_rows if row["source_type"] == "crafted"]
    figure_width = max(SINGLE_COLUMN_WIDTH, 4.55)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, 2.18),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.93, wspace=0.42)
    panels = (
        (axes[0], public_rows, _PUBLIC_PANEL_COLORS),
        (axes[1], crafted_rows, _CRAFTED_PANEL_COLORS),
    )
    for axis, rows, palette in panels:
        labels = [_inventory_figure_source_label(str(row["dataset_label"])) for row in rows]
        counts = [int(row["record_count"]) for row in rows]
        positions = list(range(len(labels)))
        axis.barh(positions, counts, color=palette[: len(labels)], edgecolor="white", linewidth=0.8)
        axis.set_yticks(positions, labels)
        axis.grid(axis="x", alpha=0.18, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#9FB3C8")
        axis.spines["bottom"].set_color("#9FB3C8")
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)
        axis.invert_yaxis()
        _annotate_hbar_end(axis, counts, positions)
        axis.tick_params(axis="x", labelsize=8.7)
        axis.tick_params(axis="y", labelsize=8.8)
    _save_plot(fig, output_root)
    plt.close(fig)


def _compact_slice_composition_figure(plt: Any, compact_rows: list[dict[str, Any]], output_root: Path) -> None:
    labels = [_compact_figure_source_label(str(row["dataset_label"])) for row in compact_rows]
    counts = [int(row["record_count"]) for row in compact_rows]
    colors = [_SOURCE_TYPE_COLORS[str(row["source_type"])] for row in compact_rows]
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.95), constrained_layout=True)
    ax.hlines(positions, [0.0] * len(counts), counts, color="#D7DDE5", linewidth=2.6, zorder=1)
    ax.scatter(counts, positions, s=64, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    ax.set_yticks(positions, labels)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    right_limit = max(300, (((max(counts, default=0) + 10) // 50) + 1) * 50)
    ax.set_xlim(-5, right_limit)
    ax.set_xlabel("Active compact records")
    ax.invert_yaxis()
    _annotate_hbar_end(ax, counts, positions)
    _save_plot(fig, output_root)
    plt.close(fig)


def _language_distribution_figure(plt: Any, compact_rows: list[dict[str, Any]], output_root: Path) -> None:
    languages = list(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES)
    matrix = [
        [int(dict(row["language_counts"]).get(language, 0)) for language in languages]
        for row in compact_rows
    ]
    _draw_count_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_compact_figure_source_label(str(row["dataset_label"])) for row in compact_rows],
        col_labels=[language.title() for language in languages],
        title="",
        subtitle="",
        output_root=output_root,
        cmap=_LANGUAGE_HEATMAP,
        colorbar_label="Active compact records",
    )


def _task_category_distribution_figure(plt: Any, category_rows: list[dict[str, Any]], output_root: Path) -> None:
    crafted_rows = [row for row in category_rows if row["slug"] in {"crafted_original", "crafted_translation", "crafted_stress"}]
    totals = Counter()
    for row in crafted_rows:
        totals[str(row["category_label"])] += int(row["record_count"])
    top_categories = [category for category, _ in totals.most_common(4)]
    columns = top_categories + ["Other categories"]
    dataset_order = ["Crafted Original", "Crafted Translation", "Crafted Stress"]
    matrix: list[list[int]] = []
    for dataset_label in dataset_order:
        dataset_rows = [row for row in crafted_rows if row["dataset_label"] == dataset_label]
        counts = {str(row["category_label"]): int(row["record_count"]) for row in dataset_rows}
        other_count = sum(value for category, value in counts.items() if category not in top_categories)
        matrix.append([counts.get(category, 0) for category in top_categories] + [other_count])
    _draw_count_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_source_short_label(label) for label in dataset_order],
        col_labels=[_category_short_label(label) for label in columns],
        title="",
        subtitle="",
        output_root=output_root,
        cmap=_CATEGORY_HEATMAP,
        colorbar_label="Compact crafted records",
    )


def _crafted_family_coverage_figure(plt: Any, family_rows: list[dict[str, Any]], output_root: Path) -> None:
    crafted_rows = [row for row in family_rows if row["source_type"] == "crafted"]
    totals = Counter()
    for row in crafted_rows:
        totals[str(row["template_family_label"])] += int(row["record_count"])
    top_families = [family for family, _ in totals.most_common(4)]
    columns = top_families + ["Other families"]
    dataset_order = ["Crafted Original", "Crafted Translation", "Crafted Stress"]
    matrix: list[list[int]] = []
    for dataset_label in dataset_order:
        dataset_rows = [row for row in crafted_rows if row["dataset_label"] == dataset_label]
        counts = {str(row["template_family_label"]): int(row["record_count"]) for row in dataset_rows}
        other_count = sum(value for family, value in counts.items() if family not in top_families)
        matrix.append([counts.get(family, 0) for family in top_families] + [other_count])
    _draw_count_heatmap(
        plt,
        matrix=matrix,
        row_labels=[_source_short_label(label) for label in dataset_order],
        col_labels=[_family_short_label(label) for label in columns],
        title="",
        subtitle="",
        output_root=output_root,
        cmap=_FAMILY_HEATMAP,
        colorbar_label="Compact crafted records",
    )


def _evaluation_dimensions_overview_figure(plt: Any, output_root: Path) -> None:
    import math

    figure_width = max(SINGLE_COLUMN_WIDTH, 4.45)
    fig = plt.figure(figsize=(figure_width, 2.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.35])
    radar = fig.add_subplot(grid[0, 0], polar=True)
    formula_ax = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.15, wspace=0.16)

    labels = ["Detection", "Robustness", "Utility", "Stealth", "Generalization"]
    angles = [index / float(len(labels)) * 2.0 * math.pi for index in range(len(labels))]
    angles += angles[:1]
    radar.set_theta_offset(math.pi / 2.0)
    radar.set_theta_direction(-1)
    radar.set_xticks(angles[:-1], labels)
    radar.tick_params(axis="x", pad=12, labelsize=7.2)
    radar.set_yticks([0.25, 0.5, 0.75], [])
    radar.set_ylim(0.0, 1.0)
    radar.grid(color="#D7DDE5", linewidth=0.8)
    radar.spines["polar"].set_color("#C7D2DE")
    radar.spines["polar"].set_linewidth(0.9)

    formula_ax.set_xlim(0.0, 1.0)
    formula_ax.set_ylim(0.0, 1.0)
    formula_ax.axis("off")
    edge_color = "#B8C7D9"

    def _tree_label(x: float, y: float, text: str, *, size: float, weight: str = "normal", color: str = "#102A43") -> None:
        formula_ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)

    def _tree_link(x0: float, y0: float, x1: float, y1: float, *, lw: float = 1.2) -> None:
        formula_ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=lw, solid_capstyle="round")

    root = (0.52, 0.84)
    base = (0.26, 0.60)
    gate = (0.82, 0.60)
    base_leaves = [
        (0.08, 0.36, "Det"),
        (0.18, 0.36, "Rob"),
        (0.28, 0.36, "Util"),
        (0.38, 0.36, "St"),
        (0.48, 0.36, "Gen"),
    ]
    gate_leaves = [
        (0.62, 0.36, "Pass\nretention"),
        (0.82, 0.36, "1 - neg.\nFPR"),
        (0.98, 0.36, "Neg.\nsupport"),
    ]

    _tree_label(*root, "CodeWMScore", size=10.8, weight="semibold")
    _tree_label(*base, "Base", size=9.9)
    _tree_label(*gate, "Gate", size=9.9)
    _tree_link(root[0], root[1] - 0.03, base[0], base[1] + 0.04, lw=1.4)
    _tree_link(root[0], root[1] - 0.03, gate[0], gate[1] + 0.04, lw=1.4)
    for child_x, child_y, label in base_leaves:
        _tree_label(child_x, child_y, label, size=8.1, color="#243B53")
        _tree_link(base[0], base[1] - 0.04, child_x, child_y + 0.03)
    for child_x, child_y, label in gate_leaves:
        _tree_label(child_x, child_y, label, size=6.5, color="#243B53")
        _tree_link(gate[0], gate[1] - 0.04, child_x, child_y + 0.03)
    formula_ax.plot([0.02, 0.98], [0.21, 0.21], color="#D9E2EC", linewidth=0.8)
    formula_ax.text(0.02, 0.10, "Det = detection reliability", ha="left", va="center", fontsize=7.4, color="#486581")
    formula_ax.text(0.56, 0.10, "Rob = robustness", ha="left", va="center", fontsize=7.4, color="#486581")
    formula_ax.text(0.02, 0.03, "Util = utility", ha="left", va="center", fontsize=7.4, color="#486581")
    formula_ax.text(0.42, 0.03, "St = stealth", ha="left", va="center", fontsize=7.4, color="#486581")
    formula_ax.text(0.71, 0.03, "Gen = generalization", ha="left", va="center", fontsize=7.4, color="#486581")

    _save_plot(fig, output_root)
    plt.close(fig)


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scope": row["scope"],
            "slug": row["slug"],
            "dataset_label": row["dataset_label"],
            "source_group": row["source_group"],
            "source_type": row["source_type"],
            "aggregate_score": row["aggregate_score"],
            "execution_slice": row["execution_slice"],
            "active_execution_slice": row["active_execution_slice"],
            "sampling_rule": row["sampling_rule"],
            "scoring_status": row["scoring_status"],
            "record_count": row["record_count"],
            "language_count": row["language_count"],
            "languages": ",".join(row["languages"]),
            "family_count": row["family_count"],
            "category_count": row["category_count"],
            "validation_scope": row["validation_scope"],
            "path": row["path"],
        }
        for row in records
    ]


def _benchmark_definition_rows(
    inventory_records: list[dict[str, Any]],
    compact_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inventory_by_slug = {row["slug"]: row for row in inventory_records}
    compact_by_slug = {row["slug"]: row for row in compact_records}
    rows: list[dict[str, Any]] = []
    for slug, inventory in inventory_by_slug.items():
        compact = compact_by_slug.get(slug)
        compact_count = int(compact["record_count"]) if compact is not None else 0
        compact_languages = ",".join(compact["languages"]) if compact is not None else ""
        execution_slice = str(compact["execution_slice"]) if compact is not None else "inventory only"
        source_label = inventory["dataset_label"]
        if slug == "humaneval_x":
            source_label = "HumanEval-X (5-lang inv.; py/cpp/java exec.)"
        elif slug == "mbxp_5lang":
            source_label = "MBXP-5lang (5-lang inv.; py/cpp/java exec.)"
        rows.append(
            {
                "source": source_label,
                "slug": slug,
                "source_group": inventory["source_group"],
                "source_type": inventory["source_type"],
                "inventory_record_count": int(inventory["record_count"]),
                "compact_record_count": compact_count,
                "scored_in_aggregate": bool(compact["aggregate_score"]) if compact is not None else False,
                "execution_slice": execution_slice,
                "inventory_languages": ",".join(inventory["languages"]),
                "execution_languages": compact_languages,
                "sampling_rule": str(compact["sampling_rule"]) if compact is not None else str(inventory["sampling_rule"]),
                "inventory_status": inventory["scoring_status"],
            }
        )
    return rows


def _language_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for language, count in sorted(record["language_counts"].items()):
            rows.append(
                {
                    "scope": record["scope"],
                    "slug": record["slug"],
                    "dataset_label": record["dataset_label"],
                    "language": language,
                    "record_count": int(count),
                }
            )
    return rows


def _category_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["source_type"] != "crafted":
            continue
        labels = dict(record.get("category_labels", {}))
        for category, count in sorted(record["category_counts"].items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "analysis_view": "crafted_only",
                    "scope": record["scope"],
                    "slug": record["slug"],
                    "dataset_label": record["dataset_label"],
                    "category": category,
                    "category_label": labels.get(category, category),
                    "record_count": int(count),
                }
            )
    return rows


def _family_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["source_type"] != "crafted":
            continue
        labels = dict(record.get("template_family_labels", {}))
        for family, count in sorted(record["template_family_counts"].items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "analysis_view": "crafted_only",
                    "scope": record["scope"],
                    "slug": record["slug"],
                    "dataset_label": record["dataset_label"],
                    "source_type": record["source_type"],
                    "template_family": family,
                    "template_family_label": labels.get(family, family),
                    "record_count": int(count),
                    "family_count": int(record["family_count"]),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    inventory_records = [_spec_summary(spec) for spec in INVENTORY_SPECS]
    compact_records = [_spec_summary(spec) for spec in COMPACT_SPECS]

    inventory_rows = _summary_rows(inventory_records)
    compact_rows = _summary_rows(compact_records)
    benchmark_definition_rows = _benchmark_definition_rows(inventory_records, compact_records)
    language_rows = _language_rows(inventory_records + compact_records)
    category_rows = _category_rows(inventory_records + compact_records)
    family_rows = _family_rows(inventory_records + compact_records)

    _write_json_csv(args.table_dir / "dataset_inventory_summary", inventory_rows)
    _write_json_csv(args.table_dir / "compact_slice_summary", compact_rows)
    _write_json_csv(args.table_dir / "benchmark_definition_summary", benchmark_definition_rows)
    _write_json_csv(args.table_dir / "dataset_language_breakdown", language_rows)
    _write_json_csv(args.table_dir / "dataset_task_category_breakdown", category_rows)
    _write_json_csv(args.table_dir / "dataset_family_breakdown", family_rows)

    _, plt = configure_matplotlib(require_times_new_roman=args.require_times_new_roman)
    _inventory_source_counts_figure(plt, inventory_rows, args.figure_dir / "inventory_source_counts")
    _compact_slice_composition_figure(plt, compact_rows, args.figure_dir / "compact_slice_composition")
    _language_distribution_figure(plt, compact_records, args.figure_dir / "language_distribution")
    _task_category_distribution_figure(plt, [row for row in category_rows if row["scope"] == "compact"], args.figure_dir / "task_category_distribution")
    _crafted_family_coverage_figure(plt, [row for row in family_rows if row["scope"] == "compact"], args.figure_dir / "crafted_family_coverage")
    _evaluation_dimensions_overview_figure(plt, args.figure_dir / "evaluation_dimensions_overview")

    summary_payload = {
        "inventory_sources": len(inventory_rows),
        "compact_sources": len(compact_rows),
        "table_dir": _display_path(args.table_dir),
        "figure_dir": _display_path(args.figure_dir),
    }
    dump_json(args.table_dir / "dataset_statistics_manifest.json", summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
