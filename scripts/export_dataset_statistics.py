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
    language_counts, language_labels = _count_field(rows, "language")
    category_counts, category_labels = _count_field(rows, "category")
    family_ids = {str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip()}
    template_family_counts, template_family_labels = _count_field(rows, "template_family")
    return {
        "slug": spec["slug"],
        "dataset_label": spec["dataset_label"],
        "source_group": spec["source_group"],
        "scope": spec["scope"],
        "source_type": spec["source_type"],
        "aggregate_score": bool(spec.get("aggregate_score", False)),
        "execution_slice": str(spec.get("execution_slice", "")),
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


def _bar_labels(values: list[str]) -> list[str]:
    mapping = {
        "HumanEval": "HE",
        "HumanEval+": "HE+",
        "MBPP+": "MBPP+",
        "HumanEval-X": "HEX",
        "HumanEval-X (py/cpp/java slice)": "HEX\npy/cpp/java",
        "MBXP-5lang": "MBXP",
        "MBXP-5lang (py/cpp/java slice)": "MBXP\npy/cpp/java",
        "Crafted Original": "Crafted\nOrig.",
        "Crafted Translation": "Crafted\nTrans.",
        "Crafted Stress": "Crafted\nStress",
    }
    return [mapping.get(value, value) for value in values]


def _save_plot(fig: Any, output_root: Path) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_root.with_suffix(".png"), bbox_inches="tight")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _inventory_source_counts_figure(plt: Any, inventory_rows: list[dict[str, Any]], output_root: Path) -> None:
    labels = [row["dataset_label"] for row in inventory_rows]
    counts = [int(row["record_count"]) for row in inventory_rows]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.8))
    ax.bar(range(len(labels)), counts, color="#355C7D")
    ax.set_title("Dataset Inventory")
    ax.set_ylabel("Records")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(_bar_labels(labels), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    _save_plot(fig, output_root)
    plt.close(fig)


def _compact_slice_composition_figure(plt: Any, compact_rows: list[dict[str, Any]], output_root: Path) -> None:
    labels = [row["dataset_label"] for row in compact_rows]
    counts = [int(row["record_count"]) for row in compact_rows]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.8))
    ax.bar(range(len(labels)), counts, color="#2A9D8F")
    ax.set_title("TOSEM-Compact Slice")
    ax.set_ylabel("Records")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(_bar_labels(labels), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    _save_plot(fig, output_root)
    plt.close(fig)


def _language_distribution_figure(plt: Any, compact_rows: list[dict[str, Any]], output_root: Path) -> None:
    languages = ["python", "cpp", "java", "javascript", "go"]
    palette = {
        "python": "#355C7D",
        "cpp": "#6C5B7B",
        "java": "#C06C84",
        "javascript": "#F4A261",
        "go": "#2A9D8F",
    }
    labels = [row["dataset_label"] for row in compact_rows]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.0))
    bottoms = [0] * len(compact_rows)
    for language in languages:
        values = [int(dict(row["language_counts"]).get(language, 0)) for row in compact_rows]
        ax.bar(
            range(len(compact_rows)),
            values,
            bottom=bottoms,
            label=language,
            color=palette[language],
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_title("Compact Language Distribution")
    ax.set_ylabel("Records")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(_bar_labels(labels), rotation=35, ha="right")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    _save_plot(fig, output_root)
    plt.close(fig)


def _task_category_distribution_figure(plt: Any, category_rows: list[dict[str, Any]], output_root: Path) -> None:
    top_rows = sorted(category_rows, key=lambda row: (-int(row["record_count"]), row["category"]))[:10]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.3))
    labels = [str(row["category_label"]) for row in top_rows]
    counts = [int(row["record_count"]) for row in top_rows]
    ax.barh(range(len(labels)), counts, color="#8D6A9F")
    ax.set_title("Compact Task Categories")
    ax.set_xlabel("Records")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    _save_plot(fig, output_root)
    plt.close(fig)


def _crafted_family_coverage_figure(plt: Any, family_rows: list[dict[str, Any]], output_root: Path) -> None:
    crafted = [row for row in family_rows if row["source_type"] == "crafted" and row["family_count"] > 0]
    labels = [str(row["dataset_label"]) for row in crafted]
    family_counts = [int(row["family_count"]) for row in crafted]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.6))
    ax.bar(range(len(labels)), family_counts, color="#E76F51")
    ax.set_title("Crafted Family Coverage")
    ax.set_ylabel("Families")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(_bar_labels(labels), rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    _save_plot(fig, output_root)
    plt.close(fig)


def _evaluation_dimensions_overview_figure(plt: Any, output_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.1))
    ax.axis("off")
    boxes = [
        (0.08, 0.72, 0.36, 0.16, "Detection\nReliability"),
        (0.56, 0.72, 0.28, 0.16, "Robustness"),
        (0.08, 0.45, 0.36, 0.16, "Utility"),
        (0.56, 0.45, 0.28, 0.16, "Stealth"),
        (0.30, 0.18, 0.34, 0.16, "Generalization"),
        (0.30, 0.01, 0.34, 0.12, "CodeWMScore"),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#F7F4EA", edgecolor="#2F4858", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
    arrowprops = dict(arrowstyle="->", color="#2F4858", linewidth=1.0)
    ax.annotate("", xy=(0.47, 0.77), xytext=(0.44, 0.77), arrowprops=arrowprops)
    ax.annotate("", xy=(0.47, 0.53), xytext=(0.44, 0.53), arrowprops=arrowprops)
    ax.annotate("", xy=(0.47, 0.26), xytext=(0.26, 0.45), arrowprops=arrowprops)
    ax.annotate("", xy=(0.47, 0.26), xytext=(0.70, 0.45), arrowprops=arrowprops)
    ax.annotate("", xy=(0.47, 0.13), xytext=(0.47, 0.18), arrowprops=arrowprops)
    ax.text(0.50, 0.145, "base score x gate", ha="center", va="bottom", fontsize=10)
    ax.set_title("Evaluation Dimensions")
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
        labels = dict(record.get("category_labels", {}))
        for category, count in sorted(record["category_counts"].items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
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
        labels = dict(record.get("template_family_labels", {}))
        for family, count in sorted(record["template_family_counts"].items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
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
    language_rows = _language_rows(inventory_records + compact_records)
    category_rows = _category_rows(inventory_records + compact_records)
    family_rows = _family_rows(inventory_records + compact_records)

    _write_json_csv(args.table_dir / "dataset_inventory_summary", inventory_rows)
    _write_json_csv(args.table_dir / "compact_slice_summary", compact_rows)
    _write_json_csv(args.table_dir / "dataset_language_breakdown", language_rows)
    _write_json_csv(args.table_dir / "dataset_task_category_breakdown", category_rows)
    _write_json_csv(args.table_dir / "dataset_family_breakdown", family_rows)

    _, plt = configure_matplotlib(require_times_new_roman=args.require_times_new_roman)
    _inventory_source_counts_figure(plt, inventory_rows, args.figure_dir / "inventory_source_counts")
    _compact_slice_composition_figure(plt, compact_rows, args.figure_dir / "compact_slice_composition")
    _language_distribution_figure(plt, compact_records, args.figure_dir / "language_distribution")
    _task_category_distribution_figure(plt, [row for row in category_rows if row["scope"] == "compact"], args.figure_dir / "task_category_distribution")
    _crafted_family_coverage_figure(plt, compact_records, args.figure_dir / "crafted_family_coverage")
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
