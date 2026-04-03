from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FIGURE_PLACEHOLDER = """# Full-Run Summary Figures

This directory is the repository-tracked export target for full-run summary figures.

In a fresh public clone, this directory may contain only this README until:

- a released raw full-run artifact is restored under `results/matrix/`, or
- the compact full matrix is rerun locally and the figure export scripts are executed.

Regenerate figures with:

```bash
python scripts/render_paper_figures.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --suite all --paper-track generation_time --require-times-new-roman --output-dir results/figures/suite_all_models_methods
```
"""

TABLE_PLACEHOLDER = """# Full-Run Summary Tables

This directory is the repository-tracked export target for full-run summary tables.

In a fresh public clone, this directory may contain only this README until:

- a released raw full-run artifact is restored under `results/matrix/`, or
- the compact full matrix is rerun locally and the table export scripts are executed.

Regenerate tables with:

```bash
python scripts/export_full_run_tables.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --output-dir results/tables/suite_all_models_methods
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete active suite outputs before a clean rerun.")
    parser.add_argument(
        "--include-full-matrix",
        action="store_true",
        help="Also remove active full-matrix outputs if present.",
    )
    parser.add_argument(
        "--include-release-bundle",
        action="store_true",
        help="Also remove the staged release bundle before the next clean rerun.",
    )
    return parser.parse_args()


def _active_paths(*, include_full_matrix: bool, include_release_bundle: bool) -> list[Path]:
    paths = [
        ROOT / "results" / "archive",
        ROOT / "results" / "audits",
        ROOT / "results" / "matrix" / "suite_canary_heavy",
        ROOT / "results" / "matrix" / "model_invocation_smoke",
        ROOT / "results" / "matrix" / "suite_all_models_methods" / "matrix_index.dry_run.json",
        ROOT / "results" / "figures" / "suite_precheck",
        ROOT / "results" / "tables" / "suite_precheck",
        ROOT / "results" / "certifications" / "suite_precheck_gate.local.json",
        ROOT / "results" / "certifications" / "suite_precheck_gate.json",
        ROOT / "results" / "certifications" / "suite_precheck.nohup.log",
        ROOT / "results" / "certifications" / "suite_precheck.live.log",
        ROOT / "results" / "certifications" / "suite_precheck.launch.json",
        ROOT / "results" / "fetched_suite",
    ]
    if include_full_matrix:
        paths.extend(
            [
                ROOT / "results" / "matrix" / "suite_all_models_methods",
                ROOT / "results" / "figures" / "suite_all_models_methods",
                ROOT / "results" / "tables" / "suite_all_models_methods",
                ROOT / "results" / "certifications" / "suite_all_models_methods_gate.json",
                ROOT / "results" / "certifications" / "suite_all_models_methods.nohup.log",
                ROOT / "results" / "certifications" / "suite_all_models_methods.launch.json",
                ROOT / "results" / "certifications" / "suite_all_models_methods.monitor.txt",
                ROOT / "results" / "matrix" / "suite_all_models_methods" / ".matrix_runner.lock",
            ]
        )
    if include_release_bundle:
        paths.append(ROOT / "results" / "release_bundle")
    paths.extend(sorted((ROOT / "results").glob("test_release_bundle*")))
    return paths


def _delete_path(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return {"deleted": str(path)}


def _reset_export_target(directory: Path, *, placeholder_name: str, placeholder_text: str) -> dict[str, str]:
    if directory.exists():
        for child in directory.iterdir():
            if child.name == placeholder_name:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    placeholder_path = directory / placeholder_name
    placeholder_path.write_text(placeholder_text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return {"reset": str(directory), "placeholder": str(placeholder_path)}


def main() -> int:
    args = parse_args()
    deleted: list[dict[str, str]] = []
    for path in _active_paths(
        include_full_matrix=args.include_full_matrix,
        include_release_bundle=args.include_release_bundle,
    ):
        item = _delete_path(path)
        if item is not None:
            deleted.append(item)
    for directory in (
        ROOT / "results",
        ROOT / "results" / "figures",
        ROOT / "results" / "tables",
        ROOT / "results" / "matrix",
        ROOT / "results" / "certifications",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _reset_export_target(
        ROOT / "results" / "figures" / "suite_all_models_methods",
        placeholder_name="README.md",
        placeholder_text=FIGURE_PLACEHOLDER,
    )
    _reset_export_target(
        ROOT / "results" / "tables" / "suite_all_models_methods",
        placeholder_name="README.md",
        placeholder_text=TABLE_PLACEHOLDER,
    )
    print(json.dumps({"deleted_count": len(deleted), "deleted": deleted}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
