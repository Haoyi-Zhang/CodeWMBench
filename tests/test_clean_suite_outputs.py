from __future__ import annotations

import shutil
import sys
from pathlib import Path

from scripts import clean_suite_outputs


ROOT = Path(__file__).resolve().parents[1]


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stale\n", encoding="utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".keep").write_text("stale\n", encoding="utf-8")


def test_clean_suite_outputs_removes_active_suite_paths_and_recreates_skeleton(monkeypatch) -> None:
    targets = [
        ROOT / "results" / "matrix" / "suite_canary_heavy",
        ROOT / "results" / "matrix" / "model_invocation_smoke",
        ROOT / "results" / "matrix" / "suite_all_models_methods",
        ROOT / "results" / "figures" / "suite_precheck",
        ROOT / "results" / "release_bundle",
    ]
    file_targets = [
        ROOT / "results" / "certifications" / "suite_precheck_gate.json",
        ROOT / "results" / "certifications" / "suite_precheck.nohup.log",
        ROOT / "results" / "certifications" / "suite_precheck.live.log",
    ]
    try:
        for path in targets:
            _ensure_dir(path)
        for path in file_targets:
            _ensure_file(path)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "clean_suite_outputs.py",
                "--include-full-matrix",
                "--include-release-bundle",
            ],
        )

        assert clean_suite_outputs.main() == 0
        for path in targets + file_targets:
            assert not path.exists()
        assert (ROOT / "results" / "matrix").exists()
        assert (ROOT / "results" / "figures").exists()
        assert (ROOT / "results" / "certifications").exists()
    finally:
        for path in targets:
            if path.exists():
                shutil.rmtree(path)
        for path in file_targets:
            if path.exists():
                path.unlink()
