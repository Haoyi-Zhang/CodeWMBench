from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import run_full_matrix


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(path: Path, *, config: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "runs": [
                    {
                        "run_id": "planner_smoke",
                        "profile": "suite_all_models_methods",
                        "config": str(config),
                        "resource": "cpu",
                        "baseline_eval": False,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_full_matrix_dry_run_writes_sidecar_index(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, config=ROOT / "configs" / "debug.yaml")
    output_root = tmp_path / "matrix"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_all_models_methods",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )

    assert run_full_matrix.main() == 0
    assert (output_root / "suite_all_models_methods" / "matrix_index.dry_run.json").exists()
    assert not (output_root / "suite_all_models_methods" / "matrix_index.json").exists()


def test_run_full_matrix_records_invalid_metadata_in_matrix_index(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, config=tmp_path / "missing-config.yaml")
    output_root = tmp_path / "matrix"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_all_models_methods",
            "--output-root",
            str(output_root),
        ],
    )

    assert run_full_matrix.main() == 1
    payload = json.loads((output_root / "suite_all_models_methods" / "matrix_index.json").read_text(encoding="utf-8"))
    assert payload["failed_count"] == 1
    assert payload["runs"][0]["reason"] == "invalid_run_metadata"


def test_run_full_matrix_cleans_stale_final_outputs_before_rerun(tmp_path: Path) -> None:
    run = run_full_matrix.MatrixRun(
        run_id="cleanup_smoke",
        config_path=ROOT / "configs" / "debug.yaml",
        resource="cpu",
        output_dir=tmp_path / "cleanup_smoke",
        report_path=tmp_path / "cleanup_smoke" / "report.json",
        log_path=tmp_path / "cleanup_smoke" / "run.log",
    )
    for name in ("report.json", "baseline_eval.json", "analysis.json"):
        path = run.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")

    run_full_matrix._cleanup_previous_final_outputs(run)

    assert not (run.output_dir / "report.json").exists()
    assert not (run.output_dir / "baseline_eval.json").exists()
    assert not (run.output_dir / "analysis.json").exists()
