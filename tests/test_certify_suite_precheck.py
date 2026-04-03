from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import certify_suite_precheck


def test_first_report_path_reads_matrix_index(tmp_path: Path) -> None:
    report_path = tmp_path / "run" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {"report_path": ""},
                    {"report_path": str(report_path)},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = certify_suite_precheck._first_report_path(matrix_index)

    assert resolved == report_path


def test_render_command_requires_times_new_roman_and_anchor_report(tmp_path: Path) -> None:
    matrix_index = tmp_path / "matrix_index.json"
    anchor_report = tmp_path / "report.json"
    output_dir = tmp_path / "figures"

    command = certify_suite_precheck._render_command(
        python_bin="python",
        matrix_index_path=matrix_index,
        anchor_report_path=anchor_report,
        figure_output_dir=output_dir,
    )

    assert "--matrix-index" in command
    assert str(matrix_index) in command
    assert "--anchor-report" in command
    assert str(anchor_report) in command
    assert "--require-times-new-roman" in command
    assert "--suite" in command
    assert "all" in command


def test_main_fails_when_render_step_fails(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=tmp_path / "results" / "matrix",
        figure_output_dir=tmp_path / "results" / "figures",
        output=output_path,
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=False,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)
    monkeypatch.setattr(certify_suite_precheck, "_first_report_path", lambda _: tmp_path / "anchor_report.json")

    def fake_run_step(command, *, timeout_seconds, label):
        return {
            "name": label,
            "command": command,
            "returncode": 1 if label == "render_suite_figures" else 0,
            "duration_seconds": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
            "status": "failed" if label == "render_suite_figures" else "passed",
        }

    monkeypatch.setattr(certify_suite_precheck, "_run_step", fake_run_step)
    monkeypatch.setattr(
        certify_suite_precheck,
        "_run_matrix_step",
        lambda command, *, timeout_seconds, label, gate_output_path, steps, matrix_index_path: fake_run_step(
            command,
            timeout_seconds=timeout_seconds,
            label=label,
        ),
    )

    exit_code = certify_suite_precheck.main()

    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["current_step"] == "render_suite_figures"
    assert payload["steps"][-1]["name"] == "render_suite_figures"


def test_precheck_uses_suite_benchmark_audit_profile(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=tmp_path / "results" / "matrix",
        figure_output_dir=tmp_path / "results" / "figures",
        output=output_path,
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=False,
        fail_fast=True,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)
    monkeypatch.setattr(certify_suite_precheck, "_first_report_path", lambda _: tmp_path / "anchor_report.json")

    seen_commands: list[tuple[str, list[str]]] = []

    def fake_run_step(command, *, timeout_seconds, label):
        seen_commands.append((label, list(command)))
        return {
            "name": label,
            "command": command,
            "returncode": 0,
            "duration_seconds": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
            "status": "passed",
        }

    monkeypatch.setattr(certify_suite_precheck, "_run_step", fake_run_step)
    monkeypatch.setattr(
        certify_suite_precheck,
        "_run_matrix_step",
        lambda command, *, timeout_seconds, label, gate_output_path, steps, matrix_index_path: fake_run_step(
            command,
            timeout_seconds=timeout_seconds,
            label=label,
        ),
    )

    exit_code = certify_suite_precheck.main()

    assert exit_code == 0
    audit_command = next(command for label, command in seen_commands if label == "audit_benchmarks")
    assert "--profile" in audit_command
    assert "suite_all_models_methods" in audit_command
    assert "--manifest" in audit_command
    assert "--matrix-profile" in audit_command
