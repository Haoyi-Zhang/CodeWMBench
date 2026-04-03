from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import export_dataset_statistics


def test_export_dataset_statistics_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_dataset_statistics.py",
            "--table-dir",
            str(table_dir),
            "--figure-dir",
            str(figure_dir),
            "--allow-font-fallback",
        ],
    )

    assert export_dataset_statistics.main() == 0

    expected_tables = {
        "dataset_inventory_summary.json",
        "compact_slice_summary.json",
        "dataset_language_breakdown.json",
        "dataset_task_category_breakdown.json",
        "dataset_family_breakdown.json",
        "dataset_statistics_manifest.json",
    }
    assert expected_tables.issubset({path.name for path in table_dir.iterdir()})

    expected_figures = {
        "inventory_source_counts.pdf",
        "compact_slice_composition.pdf",
        "language_distribution.pdf",
        "task_category_distribution.pdf",
        "crafted_family_coverage.pdf",
        "evaluation_dimensions_overview.pdf",
    }
    assert expected_figures.issubset({path.name for path in figure_dir.iterdir()})

    manifest = json.loads((table_dir / "dataset_statistics_manifest.json").read_text(encoding="utf-8"))
    assert manifest["inventory_sources"] == 8
    assert manifest["compact_sources"] == 7
