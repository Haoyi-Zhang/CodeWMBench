from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-stage CodeWMBench suite precheck.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--full-manifest", type=Path, default=Path("configs/matrices/suite_all_models_methods.json"))
    parser.add_argument("--full-profile", type=str, default="suite_all_models_methods")
    parser.add_argument("--stage-a-manifest", type=Path, default=Path("configs/matrices/suite_canary_heavy.json"))
    parser.add_argument("--stage-a-profile", type=str, default="suite_canary_heavy")
    parser.add_argument("--stage-b-manifest", type=Path, default=Path("configs/matrices/model_invocation_smoke.json"))
    parser.add_argument("--stage-b-profile", type=str, default="model_invocation_smoke")
    parser.add_argument("--output-root", type=Path, default=Path("results/matrix"))
    parser.add_argument("--figure-output-dir", type=Path, default=Path("results/figures/suite_precheck"))
    parser.add_argument("--output", type=Path, default=Path("results/certifications/suite_precheck_gate.json"))
    parser.add_argument("--gpu-slots", type=int, default=8)
    parser.add_argument("--gpu-pool-mode", choices=("split", "shared"), default="shared")
    parser.add_argument("--cpu-workers", type=int, default=12)
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--step-timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--skip-hf-access", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _run_step(command: list[str], *, timeout_seconds: int, label: str) -> dict[str, Any]:
    print(f"[suite_precheck] start {label}: {' '.join(command)}", flush=True)
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1, int(timeout_seconds)),
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")},
    )
    result = {
        "name": label,
        "command": command,
        "returncode": int(completed.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
        "status": "passed" if int(completed.returncode) == 0 else "failed",
    }
    print(
        f"[suite_precheck] finish {label}: status={result['status']} returncode={result['returncode']} duration_seconds={result['duration_seconds']}",
        flush=True,
    )
    return result


def _tail_text(path: Path, *, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    payload = path.read_text(encoding="utf-8", errors="replace")
    return payload[-limit:]


def _matrix_progress(matrix_index_path: Path) -> dict[str, Any] | None:
    if not matrix_index_path.exists():
        return None
    try:
        from monitor_matrix import build_dashboard_data

        dashboard = build_dashboard_data(matrix_index_path)
    except Exception:
        return None
    overall = dict(dashboard.get("overall", {}))
    longest_tail = dict(dashboard.get("longest_tail") or {})
    return {
        "matrix_index": str(matrix_index_path),
        "success_count": int(overall.get("success_count", 0) or 0),
        "running_count": int(overall.get("running_count", 0) or 0),
        "failed_count": int(overall.get("failed_count", 0) or 0),
        "pending_count": int(overall.get("pending_count", 0) or 0),
        "progress_fraction": float(overall.get("progress_fraction", 0.0) or 0.0),
        "eta_seconds": float(overall.get("eta_seconds", 0.0) or 0.0) if overall.get("eta_seconds") is not None else None,
        "active_models": list(dashboard.get("active_models", [])),
        "completed_models": list(dashboard.get("completed_models", [])),
        "longest_tail": {
            "run_id": str(longest_tail.get("run_id", "")),
            "model_name": str(longest_tail.get("model_name", "")),
            "benchmark_name": str(longest_tail.get("benchmark_name", "")),
            "method_name": str(longest_tail.get("method_name", "")),
            "elapsed_seconds": float(longest_tail.get("elapsed_seconds", 0.0) or 0.0),
        }
        if longest_tail
        else {},
    }


def _run_matrix_step(
    command: list[str],
    *,
    timeout_seconds: int,
    label: str,
    gate_output_path: Path,
    steps: list[dict[str, Any]],
    matrix_index_path: Path,
) -> dict[str, Any]:
    print(f"[suite_precheck] start {label}: {' '.join(command)}", flush=True)
    started = time.time()
    stdout_handle = tempfile.NamedTemporaryFile("w+", encoding="utf-8", newline="\n", delete=False)
    stderr_handle = tempfile.NamedTemporaryFile("w+", encoding="utf-8", newline="\n", delete=False)
    stdout_path = Path(stdout_handle.name)
    stderr_path = Path(stderr_handle.name)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")},
    )
    try:
        while True:
            returncode = process.poll()
            progress = _matrix_progress(matrix_index_path)
            _write_payload(
                gate_output_path,
                {
                    "status": "running",
                    "current_step": label,
                    "steps": steps,
                    "matrix_progress": progress,
                },
            )
            if returncode is not None:
                break
            if time.time() - started > max(1, int(timeout_seconds)):
                process.kill()
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            time.sleep(2.0)
    finally:
        try:
            process.wait(timeout=5)
        except Exception:
            pass
        try:
            stdout_handle.close()
        except Exception:
            pass
        try:
            stderr_handle.close()
        except Exception:
            pass
    result = {
        "name": label,
        "command": command,
        "returncode": int(process.returncode or 0),
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": _tail_text(stdout_path),
        "stderr_tail": _tail_text(stderr_path),
        "status": "passed" if int(process.returncode or 0) == 0 else "failed",
        "matrix_progress": _matrix_progress(matrix_index_path),
    }
    try:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
    except Exception:
        pass
    print(
        f"[suite_precheck] finish {label}: status={result['status']} returncode={result['returncode']} duration_seconds={result['duration_seconds']}",
        flush=True,
    )
    return result


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _matrix_index_path(output_root: Path, profile: str) -> Path:
    return output_root / profile / "matrix_index.json"


def _first_report_path(matrix_index_path: Path) -> Path:
    payload = json.loads(matrix_index_path.read_text(encoding="utf-8"))
    for run in payload.get("runs", []):
        report_path = str(run.get("report_path", "")).strip()
        if report_path:
            candidate = Path(report_path)
            if candidate.exists():
                return candidate
    raise RuntimeError(f"no report_path found in {matrix_index_path}")


def _render_command(
    *,
    python_bin: str,
    matrix_index_path: Path,
    anchor_report_path: Path,
    figure_output_dir: Path,
) -> list[str]:
    return [
        python_bin,
        str(ROOT / "scripts" / "render_paper_figures.py"),
        "--matrix-index",
        str(matrix_index_path),
        "--suite",
        "all",
        "--paper-track",
        "generation_time",
        "--anchor-report",
        str(anchor_report_path),
        "--include-reference-artifacts",
        "--require-times-new-roman",
        "--output-dir",
        str(figure_output_dir),
        "--prefix",
        "suite_precheck",
    ]


def main() -> int:
    args = parse_args()
    fail_fast = bool(getattr(args, "fail_fast", False))
    output_path = _resolve(args.output)
    output_root = _resolve(args.output_root)
    figure_output_dir = _resolve(args.figure_output_dir)
    steps: list[dict[str, Any]] = []

    stage_commands = [
        (
            "build_suite_manifests",
            [
                args.python_bin,
                str(ROOT / "scripts" / "build_suite_manifests.py"),
            ],
        ),
        (
            "audit_benchmarks",
            [
                args.python_bin,
                str(ROOT / "scripts" / "audit_benchmarks.py"),
                "--manifest",
                str(_resolve(args.full_manifest)),
                "--matrix-profile",
                args.full_profile,
                "--profile",
                args.full_profile,
            ],
        ),
        (
            "audit_suite_matrix",
            [
                args.python_bin,
                str(ROOT / "scripts" / "audit_full_matrix.py"),
                "--manifest",
                str(_resolve(args.full_manifest)),
                "--profile",
                args.full_profile,
                "--strict-hf-cache",
                "--model-load-smoke",
                "--runtime-smoke",
                *(["--skip-hf-access"] if args.skip_hf_access else []),
            ],
        ),
        (
            "stage_a_suite_canary_heavy",
            [
                args.python_bin,
                str(ROOT / "scripts" / "run_full_matrix.py"),
                "--manifest",
                str(_resolve(args.stage_a_manifest)),
                "--profile",
                args.stage_a_profile,
                "--output-root",
                str(output_root),
                "--gpu-slots",
                str(args.gpu_slots),
                "--gpu-pool-mode",
                args.gpu_pool_mode,
                "--cpu-workers",
                str(args.cpu_workers),
                "--retry-count",
                str(args.retry_count),
                *(["--fail-fast"] if fail_fast else []),
                *(["--resume"] if args.resume else []),
            ],
        ),
        (
            "stage_b_model_invocation_smoke",
            [
                args.python_bin,
                str(ROOT / "scripts" / "run_full_matrix.py"),
                "--manifest",
                str(_resolve(args.stage_b_manifest)),
                "--profile",
                args.stage_b_profile,
                "--output-root",
                str(output_root),
                "--gpu-slots",
                str(args.gpu_slots),
                "--gpu-pool-mode",
                args.gpu_pool_mode,
                "--cpu-workers",
                str(args.cpu_workers),
                "--retry-count",
                str(args.retry_count),
                *(["--fail-fast"] if fail_fast else []),
                *(["--resume"] if args.resume else []),
            ],
        ),
    ]

    status = "passed"
    for label, command in stage_commands:
        _write_payload(
            output_path,
            {
                "status": "running",
                "current_step": label,
                "steps": steps,
            },
        )
        if label == "stage_a_suite_canary_heavy":
            result = _run_matrix_step(
                command,
                timeout_seconds=args.step_timeout_seconds,
                label=label,
                gate_output_path=output_path,
                steps=steps,
                matrix_index_path=_matrix_index_path(output_root, args.stage_a_profile),
            )
        elif label == "stage_b_model_invocation_smoke":
            result = _run_matrix_step(
                command,
                timeout_seconds=args.step_timeout_seconds,
                label=label,
                gate_output_path=output_path,
                steps=steps,
                matrix_index_path=_matrix_index_path(output_root, args.stage_b_profile),
            )
        else:
            result = _run_step(command, timeout_seconds=args.step_timeout_seconds, label=label)
        steps.append(result)
        payload = {
            "status": "running" if result["status"] == "passed" else "failed",
            "current_step": label,
            "steps": steps,
        }
        if result.get("matrix_progress") is not None:
            payload["matrix_progress"] = result["matrix_progress"]
        if result["status"] != "passed":
            status = "failed"
            payload["status"] = status
            _write_payload(output_path, payload)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 1
        _write_payload(output_path, payload)

    try:
        stage_a_index = _matrix_index_path(output_root, args.stage_a_profile)
        anchor_report_path = _first_report_path(stage_a_index)
    except Exception as exc:  # pragma: no cover - defensive gate failure path
        payload = {
            "status": "failed",
            "current_step": "render_suite_figures",
            "steps": steps,
            "error": str(exc),
        }
        _write_payload(output_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    _write_payload(
        output_path,
        {
            "status": "running",
            "current_step": "render_suite_figures",
            "steps": steps,
        },
    )
    render_step = _run_step(
        _render_command(
            python_bin=args.python_bin,
            matrix_index_path=stage_a_index,
            anchor_report_path=anchor_report_path,
            figure_output_dir=figure_output_dir,
        ),
        timeout_seconds=args.step_timeout_seconds,
        label="render_suite_figures",
    )
    steps.append(render_step)
    if render_step["status"] != "passed":
        payload = {
            "status": "failed",
            "current_step": "render_suite_figures",
            "steps": steps,
        }
        _write_payload(output_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1
    _write_payload(
        output_path,
        {
            "status": "running",
            "current_step": "render_suite_figures",
            "steps": steps,
        },
    )

    final_payload = {
        "status": status,
        "current_step": "complete",
        "steps": steps,
        "stage_a_profile": args.stage_a_profile,
        "stage_b_profile": args.stage_b_profile,
        "full_profile": args.full_profile,
        "figure_output_dir": str(figure_output_dir),
        "next_command": [
            args.python_bin,
            str(ROOT / "scripts" / "run_full_matrix.py"),
            "--manifest",
            str(_resolve(args.full_manifest)),
            "--profile",
            args.full_profile,
            "--output-root",
            str(output_root),
            "--gpu-slots",
            str(args.gpu_slots),
            "--gpu-pool-mode",
            args.gpu_pool_mode,
            "--cpu-workers",
            str(args.cpu_workers),
            "--retry-count",
            str(args.retry_count),
            "--fail-fast",
        ],
    }
    _write_payload(output_path, final_payload)
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
