from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _shared import dump_json, ensure_dir, load_json
from codewmbench.config import build_experiment_config, load_config, merge_config_source


@dataclass(frozen=True, slots=True)
class MatrixRun:
    run_id: str
    config_path: Path
    resource: str
    output_dir: Path
    report_path: Path
    log_path: Path
    gpu_pool: str = ""
    baseline_eval: bool = False
    baseline_eval_sample_limit: int | None = None
    profile: str = "suite_all_models_methods"
    tags: tuple[str, ...] = ()
    config_overrides: dict[str, Any] | None = None
    priority: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CodeWMBench suite matrix.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/matrices/suite_all_models_methods.json"))
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip runs whose report already exists.")
    parser.add_argument("--gpu-slots", type=int, default=4, help="Total visible GPU slots to schedule across.")
    parser.add_argument(
        "--gpu-pool-mode",
        choices=("split", "shared"),
        default="shared",
        help="Use split per-pool GPU queues or a single shared GPU queue for all GPU runs.",
    )
    parser.add_argument("--cpu-workers", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=Path("results/matrix"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop launching additional runs after the first engineering failure; queued runs are skipped cleanly.",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=24 * 60 * 60,
        help="Hard timeout for each child command to prevent hung runs from pinning workers forever.",
    )
    return parser.parse_args()


def _load_matrix_runs(manifest_path: Path, profile: str, output_root: Path) -> tuple[dict[str, Any], list[MatrixRun]]:
    manifest = load_json(manifest_path)
    runs: list[MatrixRun] = []
    for item in manifest.get("runs", []):
        if str(item.get("profile", profile)) != profile:
            continue
        run_id = str(item["run_id"])
        config_path = Path(str(item["config"]))
        output_dir = output_root / profile / run_id
        resource = str(item.get("resource", "cpu")).lower()
        if resource not in {"cpu", "gpu"}:
            raise ValueError(f"unsupported matrix resource '{resource}' in {manifest_path}; API support removed")
        runs.append(
            MatrixRun(
                run_id=run_id,
                config_path=config_path,
                resource=resource,
                output_dir=output_dir,
                report_path=output_dir / "report.json",
                log_path=output_dir / "run.log",
                gpu_pool=str(item.get("gpu_pool", "")).lower(),
                baseline_eval=bool(item.get("baseline_eval", False)),
                baseline_eval_sample_limit=(
                    int(item["baseline_eval_sample_limit"])
                    if item.get("baseline_eval_sample_limit") is not None
                    else None
                ),
                profile=str(item.get("profile", profile)),
                tags=tuple(str(tag) for tag in item.get("tags", [])),
                config_overrides=dict(item.get("config_overrides", {})) if isinstance(item.get("config_overrides"), dict) else {},
                priority=int(item.get("priority", 0) or 0),
            )
        )
    return manifest, runs


def _resolved_profile(manifest: dict[str, Any], requested_profile: str | None) -> str:
    profile = str(requested_profile or manifest.get("profile", "suite_all_models_methods")).strip()
    if not profile:
        return "suite_all_models_methods"
    return profile


def _slot_mapping(total_slots: int, pool_mode: str = "split") -> dict[str, list[int]]:
    total = max(1, int(total_slots))
    slots = list(range(total))
    if str(pool_mode).lower() == "shared":
        return {"default": slots}
    if total == 1:
        return {"default": slots, "runtime": slots, "local_hf": slots}
    runtime_count = max(1, total // 2)
    runtime_slots = slots[:runtime_count]
    local_slots = slots[runtime_count:] or runtime_slots
    return {"default": slots, "runtime": runtime_slots, "local_hf": local_slots}


def _executor_specs(slot_mapping: dict[str, list[int]], *, cpu_workers: int) -> dict[str, int]:
    specs = {
        "cpu": max(1, int(cpu_workers)),
    }
    for pool_name, pool_slots in slot_mapping.items():
        specs[f"gpu:{pool_name}"] = max(1, len(pool_slots))
    return specs


def _executor_key_for(run: MatrixRun, slot_mapping: dict[str, list[int]]) -> str:
    if run.resource != "gpu":
        return "cpu"
    pool_name = run.gpu_pool if run.gpu_pool in slot_mapping else "default"
    return f"gpu:{pool_name}"


def _resolved_config_source(run: MatrixRun) -> dict[str, Any]:
    return merge_config_source(load_config(run.config_path), **dict(run.config_overrides or {}))


def _load_run_metadata(config_path: Path) -> dict[str, Any]:
    config = build_experiment_config(load_config(config_path))
    provider_model = str(config.provider_parameters.get("model", "")).strip()
    runtime_model = str(config.metadata.get("watermark", {}).get("model_name", "")).strip()
    return {
        "corpus_size": config.corpus_size,
        "provider_mode": config.provider_mode,
        "watermark_name": config.watermark_name,
        "provider_model": provider_model,
        "runtime_model": runtime_model,
        "effective_model": provider_model or runtime_model,
        "benchmark_label": str(config.corpus_parameters.get("dataset_label", "")).strip(),
        "benchmark_path": str(config.corpus_parameters.get("prepared_benchmark", "")),
    }


def _resolved_experiment_config(run: MatrixRun, *, output_path: str | None = None):
    raw = _resolved_config_source(run)
    if output_path is not None:
        raw = merge_config_source(raw, output_path=output_path)
    return build_experiment_config(raw)


def _requires_offline_hf_assets(run: MatrixRun) -> bool:
    config = _resolved_experiment_config(run)
    provider_local_only = bool(config.provider_parameters.get("local_files_only", False))
    runtime_local_only = bool(config.metadata.get("watermark", {}).get("local_files_only", False))
    return provider_local_only or runtime_local_only


def _materialize_run_config(run: MatrixRun) -> Path:
    ensure_dir(run.output_dir)
    resolved_path = run.output_dir / "_resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(_resolved_config_source(run), sort_keys=False), encoding="utf-8")
    return resolved_path


def _command_for(run: MatrixRun, python_bin: str) -> list[str]:
    resolved_config_path = _materialize_run_config(run)
    return [
        python_bin,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--config",
        str(resolved_config_path),
        "--output",
        str(run.output_dir),
    ]


def _baseline_eval_command(run: MatrixRun, python_bin: str, *, payload_path: Path | None = None) -> list[str]:
    command = [
        python_bin,
        str(ROOT / "scripts" / "evaluate_baseline_family.py"),
        "--input",
        str(run.output_dir),
        "--output",
        str(run.output_dir / "baseline_eval.json"),
    ]
    if payload_path is not None:
        command.extend(["--payloads", str(payload_path)])
    if run.baseline_eval_sample_limit is not None:
        command.extend(["--sample-limit", str(int(run.baseline_eval_sample_limit))])
    return command


def _cleanup_sensitive_baseline_payloads(path: Path | None) -> None:
    preserve = str(os.environ.get("CODEWMBENCH_PRESERVE_BASELINE_EVAL_PAYLOADS", "")).strip().lower()
    if preserve in {"1", "true", "yes", "on"}:
        return
    if path is not None and path.exists():
        path.unlink()


def _load_run_metadata_for_matrix(run: MatrixRun) -> dict[str, Any]:
    if not run.config_overrides:
        return _load_run_metadata(run.config_path)
    config = _resolved_experiment_config(run)
    provider_model = str(config.provider_parameters.get("model", "")).strip()
    runtime_model = str(config.metadata.get("watermark", {}).get("model_name", "")).strip()
    return {
        "corpus_size": config.corpus_size,
        "provider_mode": config.provider_mode,
        "watermark_name": config.watermark_name,
        "provider_model": provider_model,
        "runtime_model": runtime_model,
        "effective_model": provider_model or runtime_model,
        "benchmark_label": str(config.corpus_parameters.get("dataset_label", "")).strip(),
        "benchmark_path": str(config.corpus_parameters.get("prepared_benchmark", "")),
    }


def _append_log_message(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[matrix_runner] {message}\n")


def _expected_report_config(run: MatrixRun) -> dict[str, Any]:
    config = _resolved_experiment_config(run, output_path=str(run.report_path))
    return config.as_dict()


def _load_report_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _report_is_complete(run: MatrixRun) -> tuple[bool, str]:
    payload = _load_report_payload(run.report_path)
    if payload is None:
        return False, "invalid_or_missing_report"
    rows = payload.get("rows", [])
    summary = payload.get("summary", {})
    if not isinstance(rows, list) or not rows:
        return False, "missing_or_empty_rows"
    if not isinstance(summary, dict):
        return False, "missing_summary"
    record_count = summary.get("record_count")
    if record_count is not None:
        try:
            if int(record_count) != len(rows):
                return False, "summary_record_count_mismatch"
        except Exception:
            return False, "summary_record_count_invalid"
    report_config = payload.get("config")
    if not isinstance(report_config, dict):
        return False, "missing_report_config"
    if report_config != _expected_report_config(run):
        return False, "config_mismatch"
    return True, "ok"


def _baseline_eval_is_complete(run: MatrixRun) -> tuple[bool, str]:
    if not run.baseline_eval:
        return True, "not_required"
    baseline_eval_path = run.output_dir / "baseline_eval.json"
    if not baseline_eval_path.exists():
        return False, "missing_baseline_eval"
    try:
        payload = load_json(baseline_eval_path)
    except Exception:
        return False, "invalid_baseline_eval_json"
    if not isinstance(payload, dict):
        return False, "invalid_baseline_eval_payload"
    record_count = payload.get("record_count")
    if record_count is not None:
        try:
            if int(record_count) <= 0:
                return False, "empty_baseline_eval"
        except Exception:
            return False, "invalid_baseline_eval_record_count"
    return True, "ok"


_MODEL_SCALE_PATTERN = re.compile(r"(?P<size>\d+(?:\.\d+)?)\s*[bB]")


def _model_scale_hint(model_name: str) -> float:
    match = _MODEL_SCALE_PATTERN.search(str(model_name))
    if match is None:
        return 0.0
    try:
        return float(match.group("size"))
    except Exception:
        return 0.0


def _submission_priority(run: MatrixRun) -> tuple[int, int, float, str]:
    metadata = _load_run_metadata_for_matrix(run)
    corpus_size = int(metadata.get("corpus_size", 0) or 0)
    model_scale = _model_scale_hint(str(metadata.get("effective_model", "")))
    if run.resource == "gpu":
        runtime_rank = 1 if run.gpu_pool == "runtime" else 0
        baseline_rank = 1 if run.baseline_eval else 0
        return (3, int(run.priority), runtime_rank * 10 + baseline_rank, float(corpus_size) + model_scale, run.run_id)
    return (1, int(run.priority), 0, float(corpus_size), run.run_id)


def _submission_priority_with_metadata(run: MatrixRun, metadata: dict[str, Any]) -> tuple[int, int, float, str]:
    corpus_size = int(metadata.get("corpus_size", 0) or 0)
    model_scale = _model_scale_hint(str(metadata.get("effective_model", "")))
    if run.resource == "gpu":
        runtime_rank = 1 if run.gpu_pool == "runtime" else 0
        baseline_rank = 1 if run.baseline_eval else 0
        return (3, int(run.priority), runtime_rank * 10 + baseline_rank, float(corpus_size) + model_scale, run.run_id)
    return (1, int(run.priority), 0, float(corpus_size), run.run_id)


def _model_schedule_rank(model_name: str) -> int:
    name = str(model_name).strip().lower()
    if "qwen2.5-coder-14b" in name:
        return 0
    if "deepseek-coder-6.7b" in name:
        return 1
    if "qwen2.5-coder-7b" in name:
        return 2
    if "starcoder2-7b" in name:
        return 3
    return 999


def _method_schedule_rank(method_name: str) -> int:
    name = str(method_name).strip().lower()
    order = {
        "ewd_runtime": 0,
        "kgw_runtime": 1,
        "sweet_runtime": 2,
        "stone_runtime": 3,
    }
    return order.get(name, 999)


def _benchmark_schedule_rank(label: str) -> int:
    name = str(label).strip().lower()
    order = {
        "crafted original": 0,
        "crafted translation": 1,
        "crafted stress": 2,
        "mbxp-5lang": 3,
        "humaneval-x": 4,
        "mbpp+": 5,
        "humaneval+": 6,
    }
    return order.get(name, 999)


def _gpu_interleaved_runs(runs: list[MatrixRun], metadata_map: dict[str, dict[str, Any]]) -> list[MatrixRun]:
    gpu_runs = [run for run in runs if run.resource == "gpu"]
    cpu_runs = [run for run in runs if run.resource != "gpu"]
    buckets: dict[str, list[MatrixRun]] = {}
    for run in gpu_runs:
        metadata = metadata_map[run.run_id]
        model_name = str(metadata.get("effective_model", "")).strip() or "unknown_model"
        buckets.setdefault(model_name, []).append(run)
    ordered_models = sorted(
        buckets,
        key=lambda model_name: (
            _model_schedule_rank(model_name),
            -_model_scale_hint(model_name),
            model_name,
        ),
    )
    for model_name, bucket in buckets.items():
        bucket.sort(
            key=lambda run: (
                _method_schedule_rank(str(metadata_map[run.run_id].get("watermark_name", ""))),
                _benchmark_schedule_rank(str(metadata_map[run.run_id].get("benchmark_label", ""))),
                -int(run.priority),
                -int(metadata_map[run.run_id].get("corpus_size", 0) or 0),
                run.run_id,
            )
        )
    ordered_gpu: list[MatrixRun] = []
    while True:
        emitted = False
        for model_name in ordered_models:
            bucket = buckets.get(model_name, [])
            if bucket:
                ordered_gpu.append(bucket.pop(0))
                emitted = True
        if not emitted:
            break
    ordered_cpu = sorted(cpu_runs, key=lambda run: _submission_priority_with_metadata(run, metadata_map[run.run_id]), reverse=True)
    return ordered_gpu + ordered_cpu


def _ordered_runs(runs: list[MatrixRun], *, metadata_map: dict[str, dict[str, Any]] | None = None) -> list[MatrixRun]:
    if metadata_map is None:
        return sorted(runs, key=_submission_priority, reverse=True)
    return _gpu_interleaved_runs(runs, metadata_map)


def _write_index(index_path: Path, payload: dict[str, Any]) -> None:
    dump_json(index_path, payload)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_lock_payload(lock_path: Path, payload: dict[str, Any], *, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT
    if exclusive:
        flags |= os.O_EXCL
    else:
        flags |= os.O_TRUNC
    fd = os.open(str(lock_path), flags)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _acquire_matrix_lock(lock_path: Path, *, manifest_path: Path, profile: str) -> dict[str, Any]:
    ensure_dir(lock_path.parent)
    owner = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "profile": profile,
        "manifest": str(manifest_path),
        "created_at": time.time(),
    }
    for _ in range(2):
        try:
            _write_lock_payload(lock_path, owner, exclusive=True)
        except FileExistsError:
            existing: dict[str, Any] = {}
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            existing_host = str(existing.get("host", "")).strip()
            existing_pid = int(existing.get("pid", 0) or 0)
            if existing_host and existing_host != owner["host"]:
                raise RuntimeError(
                    f"matrix lock already held by host={existing_host} pid={existing_pid or 'unknown'} at {lock_path}"
                )
            if existing_pid > 0 and _process_alive(existing_pid):
                raise RuntimeError(f"matrix lock already held by pid={existing_pid} at {lock_path}")
            _write_lock_payload(lock_path, owner, exclusive=False)
        return owner
    raise RuntimeError(f"failed to acquire matrix lock at {lock_path}")


def _release_matrix_lock(lock_path: Path, owner: dict[str, Any] | None) -> None:
    if owner is None:
        return
    try:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
    if (
        str(existing.get("host", "")).strip() == str(owner.get("host", "")).strip()
        and int(existing.get("pid", 0) or 0) == int(owner.get("pid", 0) or 0)
    ):
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _dry_run_index_path(index_path: Path) -> Path:
    return index_path.with_name("matrix_index.dry_run.json")


def _artifact_paths_for_cleanup(run: MatrixRun) -> list[Path]:
    candidates = [
        run.report_path,
        run.output_dir / "baseline_eval.json",
        run.output_dir / "analysis.json",
    ]
    return candidates


def _cleanup_previous_final_outputs(run: MatrixRun) -> None:
    for path in _artifact_paths_for_cleanup(run):
        if path.exists():
            try:
                path.unlink()
            except FileNotFoundError:
                continue


def _blank_metadata() -> dict[str, Any]:
    return {
        "corpus_size": 0,
        "provider_mode": "",
        "watermark_name": "",
        "provider_model": "",
        "runtime_model": "",
        "effective_model": "",
        "benchmark_label": "",
        "benchmark_path": "",
    }


def _matrix_record_base(run: MatrixRun, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "run_id": run.run_id,
        "config": str(run.config_path),
        "resource": run.resource,
        "gpu_pool": run.gpu_pool,
        "output_dir": str(run.output_dir),
        "report_path": str(run.report_path),
        "log_path": str(run.log_path),
        "baseline_eval_path": str(run.output_dir / "baseline_eval.json"),
        "priority": int(run.priority),
        "tags": list(run.tags),
        "config_overrides": dict(run.config_overrides or {}),
    }
    payload.update(metadata or _blank_metadata())
    return payload


def _prevalidate_run_metadata(runs: list[MatrixRun]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    metadata_map: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for run in runs:
        try:
            metadata_map[run.run_id] = _load_run_metadata_for_matrix(run)
        except Exception as exc:
            errors[run.run_id] = f"{type(exc).__name__}: {exc}"
    return metadata_map, errors


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows-only fallback
            process.kill()
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.kill()
        except Exception:
            return


def _stream_pipe_to_log(
    pipe: Any,
    *,
    handle: Any,
    lock: threading.Lock,
) -> None:
    try:
        for chunk in iter(pipe.readline, ""):
            if not chunk:
                break
            with lock:
                handle.write(chunk)
                if not chunk.endswith("\n"):
                    handle.write("\n")
                handle.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _execute_subprocess(command: list[str], *, env: dict[str, str], log_path: Path, timeout_seconds: int) -> int:
    ensure_dir(log_path.parent)
    run_env = dict(env)
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    timeout_seconds = max(1, int(timeout_seconds))
    timed_out = False
    return_code = 0
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=run_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=(os.name != "nt"),
        )
        lock = threading.Lock()
        threads = []
        if process.stdout is not None:
            threads.append(
                threading.Thread(
                    target=_stream_pipe_to_log,
                    kwargs={"pipe": process.stdout, "handle": handle, "lock": lock},
                    daemon=True,
                )
            )
        if process.stderr is not None:
            threads.append(
                threading.Thread(
                    target=_stream_pipe_to_log,
                    kwargs={"pipe": process.stderr, "handle": handle, "lock": lock},
                    daemon=True,
                )
            )
        for thread in threads:
            thread.start()
        try:
            return_code = int(process.wait(timeout=timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process(process)
            return_code = 124
        finally:
            for thread in threads:
                thread.join(timeout=5.0)
        if timed_out:
            handle.write(f"[timeout_seconds={timeout_seconds}]\n")
        handle.write(f"[exit_code={return_code}]\n")
        handle.flush()
    return return_code


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    profile = _resolved_profile(manifest, args.profile)
    manifest, all_runs = _load_matrix_runs(args.manifest, profile, args.output_root)
    all_runs_by_id = {run.run_id: run for run in all_runs}
    output_root = args.output_root / profile
    ensure_dir(output_root)
    index_path = output_root / "matrix_index.json"
    dry_run_index_path = _dry_run_index_path(index_path)
    slot_mapping = _slot_mapping(args.gpu_slots, args.gpu_pool_mode)
    run_metadata_map, metadata_errors = _prevalidate_run_metadata(all_runs)
    runs = _ordered_runs([run for run in all_runs if run.run_id in run_metadata_map], metadata_map=run_metadata_map)
    total_run_count = len(all_runs)

    if args.dry_run:
        planned = {
            "schema_version": 1,
            "profile": profile,
            "manifest": str(args.manifest),
            "gpu_pool_mode": args.gpu_pool_mode,
            "gpu_slot_mapping": slot_mapping,
            "run_count": total_run_count,
            "planned_run_count": len(runs),
            "invalid_run_count": len(metadata_errors),
            "runs": [
                {
                    "run_id": run.run_id,
                    "config": str(run.config_path),
                    "resource": run.resource,
                    "gpu_pool": run.gpu_pool,
                    "output_dir": str(run.output_dir),
                    "report_path": str(run.report_path),
                    "log_path": str(run.log_path),
                    "priority": int(run.priority),
                    "tags": list(run.tags),
                    "config_overrides": dict(run.config_overrides or {}),
                    **run_metadata_map[run.run_id],
                }
                for run in runs
            ],
            "invalid_runs": [
                {
                    **_matrix_record_base(all_runs_by_id[run_id], metadata=_blank_metadata()),
                    "status": "failed",
                    "reason": "invalid_run_metadata",
                    "error": error,
                }
                for run_id, error in sorted(metadata_errors.items())
            ],
            "excluded_configs": list(manifest.get("excluded_configs", [])),
        }
        _write_index(dry_run_index_path, planned)
        print(json.dumps(planned, indent=2, sort_keys=True))
        return 1 if metadata_errors else 0

    lock_path = output_root / ".matrix_runner.lock"
    lock_owner: dict[str, Any] | None = None
    try:
        lock_owner = _acquire_matrix_lock(lock_path, manifest_path=args.manifest.resolve(), profile=profile)
    except RuntimeError as exc:
        print(f"matrix launch blocked: {exc}", file=sys.stderr)
        return 1

    gpu_queues: dict[str, Queue[int]] = {}
    for pool_name, pool_slots in slot_mapping.items():
        queue: Queue[int] = Queue()
        for slot in pool_slots:
            queue.put(slot)
        gpu_queues[pool_name] = queue

    results_lock = threading.Lock()
    results: dict[str, dict[str, Any]] = {}
    stop_event = threading.Event()

    def current_index() -> dict[str, Any]:
        success_count = sum(1 for item in results.values() if item["status"] == "success")
        skipped_count = sum(1 for item in results.values() if item["status"] == "skipped")
        running_count = sum(1 for item in results.values() if item["status"] == "running")
        failed_count = sum(1 for item in results.values() if item["status"] == "failed")
        return {
            "schema_version": 1,
            "profile": profile,
            "manifest": str(args.manifest),
            "gpu_pool_mode": args.gpu_pool_mode,
            "gpu_slot_mapping": slot_mapping,
            "run_count": total_run_count,
            "completed_count": success_count + skipped_count,
            "success_count": success_count,
            "skipped_count": skipped_count,
            "running_count": running_count,
            "failed_count": failed_count,
            "pending_count": max(0, total_run_count - len(results)),
            "stop_requested": stop_event.is_set(),
            "updated_at": time.time(),
            "runs": [results[key] for key in sorted(results)],
        }

    def persist_index() -> None:
        with results_lock:
            _write_index(index_path, current_index())

    def execute(run: MatrixRun) -> dict[str, Any]:
        metadata = run_metadata_map.get(run.run_id, _blank_metadata())
        if args.fail_fast and stop_event.is_set():
            record = {
                **_matrix_record_base(run, metadata=metadata),
                "status": "skipped",
                "reason": "fail_fast_stop_requested",
                "attempts": 0,
                "retries_used": 0,
                "exit_code": 0,
                "cuda_visible_devices": "",
            }
            with results_lock:
                results[run.run_id] = record
            persist_index()
            return record
        if args.resume and run.report_path.exists():
            baseline_eval_path = run.output_dir / "baseline_eval.json"
            report_complete, report_reason = _report_is_complete(run)
            baseline_complete, baseline_reason = _baseline_eval_is_complete(run)
            if report_complete and baseline_complete:
                record = {
                    **_matrix_record_base(run, metadata=metadata),
                    "baseline_eval_path": str(baseline_eval_path) if baseline_eval_path.exists() else "",
                    "status": "skipped",
                    "reason": "resume_existing_report",
                    "attempts": 0,
                    "retries_used": 0,
                    "cuda_visible_devices": "",
                }
                with results_lock:
                    results[run.run_id] = record
                persist_index()
                return record
            if not report_complete:
                _append_log_message(run.log_path, f"resume requested but existing report is not reusable: {report_reason}")
            if not baseline_complete:
                _append_log_message(run.log_path, f"resume requested but existing baseline evaluation is not reusable: {baseline_reason}")
        _cleanup_previous_final_outputs(run)

        queue_name = run.gpu_pool if run.gpu_pool in gpu_queues else "default"
        attempts_allowed = max(1, int(args.retry_count) + 1)
        last_exit_code = 0
        last_device = ""
        started_at = time.time()
        ensure_dir(run.output_dir)
        with results_lock:
            results[run.run_id] = {
                **_matrix_record_base(run, metadata=metadata),
                "status": "running",
                "attempts": 0,
                "retries_used": 0,
                "exit_code": 0,
                "started_at": started_at,
                "finished_at": None,
                "duration_seconds": 0.0,
                "cuda_visible_devices": "",
            }
        persist_index()

        for attempt in range(1, attempts_allowed + 1):
            if args.fail_fast and stop_event.is_set():
                finished_at = time.time()
                record = {
                    **_matrix_record_base(run, metadata=metadata),
                    "status": "skipped",
                    "reason": "fail_fast_stop_requested",
                    "attempts": attempt - 1,
                    "retries_used": max(0, attempt - 2),
                    "exit_code": 0,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_seconds": round(finished_at - started_at, 2),
                    "cuda_visible_devices": "",
                }
                with results_lock:
                    results[run.run_id] = record
                persist_index()
                return record
            env = dict(os.environ)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            _cleanup_previous_final_outputs(run)
            if _requires_offline_hf_assets(run):
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
            device_token = ""
            queue: Queue[int] | None = None
            device_id: int | None = None
            payload_path: Path | None = None
            if run.baseline_eval:
                handle = tempfile.NamedTemporaryFile(
                    prefix=f"{run.run_id}_baseline_eval_",
                    suffix=".private.jsonl",
                    delete=False,
                )
                handle.close()
                payload_path = Path(handle.name)
                env["CODEWMBENCH_ENABLE_BASELINE_EVAL_PAYLOADS"] = "1"
                env["CODEWMBENCH_BASELINE_EVAL_PAYLOAD_PATH"] = str(payload_path)
            if run.resource == "gpu":
                queue = gpu_queues[queue_name]
                device_id = queue.get()
                device_token = str(device_id)
                env["CUDA_VISIBLE_DEVICES"] = device_token
            with results_lock:
                running_record = dict(results.get(run.run_id, {}))
                running_record.update(
                    {
                        "status": "running",
                        "attempts": attempt,
                        "retries_used": attempt - 1,
                        "cuda_visible_devices": device_token,
                    }
                )
                results[run.run_id] = running_record
            persist_index()
            try:
                exit_code = _execute_subprocess(
                    _command_for(run, args.python_bin),
                    env=env,
                    log_path=run.log_path,
                    timeout_seconds=args.command_timeout_seconds,
                )
                baseline_eval_path = run.output_dir / "baseline_eval.json"
                if exit_code == 0 and run.report_path.exists():
                    report_complete, report_reason = _report_is_complete(run)
                    if not report_complete:
                        _append_log_message(run.log_path, f"report integrity check failed after run: {report_reason}")
                        exit_code = 91
                    if run.baseline_eval:
                        if exit_code == 0:
                            eval_code = _execute_subprocess(
                                _baseline_eval_command(run, args.python_bin, payload_path=payload_path),
                                env=env,
                                log_path=run.log_path,
                                timeout_seconds=args.command_timeout_seconds,
                            )
                            if eval_code != 0:
                                exit_code = eval_code
                        if exit_code == 0:
                            baseline_complete, baseline_reason = _baseline_eval_is_complete(run)
                            if not baseline_complete:
                                _append_log_message(
                                    run.log_path,
                                    f"baseline evaluation integrity check failed after run: {baseline_reason}",
                                )
                                exit_code = 92
                        _cleanup_sensitive_baseline_payloads(payload_path)
                    if exit_code == 0 and (not run.baseline_eval or baseline_eval_path.exists()):
                        finished_at = time.time()
                        record = {
                            **_matrix_record_base(run, metadata=metadata),
                            "baseline_eval_path": str(baseline_eval_path) if baseline_eval_path.exists() else "",
                            "status": "success",
                            "attempts": attempt,
                            "retries_used": attempt - 1,
                            "exit_code": 0,
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "duration_seconds": round(finished_at - started_at, 2),
                            "cuda_visible_devices": device_token,
                        }
                        with results_lock:
                            results[run.run_id] = record
                        persist_index()
                        return record
                last_exit_code = exit_code
                last_device = device_token
            finally:
                if run.baseline_eval:
                    _cleanup_sensitive_baseline_payloads(payload_path)
                if queue is not None and device_id is not None:
                    queue.put(device_id)
            if attempt < attempts_allowed:
                time.sleep(1.0)

        finished_at = time.time()
        record = {
            **_matrix_record_base(run, metadata=metadata),
            "status": "failed",
            "reason": "subprocess_failed",
            "attempts": attempts_allowed,
            "retries_used": attempts_allowed - 1,
            "exit_code": last_exit_code,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(finished_at - started_at, 2),
            "cuda_visible_devices": last_device,
        }
        with results_lock:
            results[run.run_id] = record
        persist_index()
        if args.fail_fast:
            stop_event.set()
        return record

    executors = {
        key: concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        for key, max_workers in _executor_specs(
            slot_mapping,
            cpu_workers=args.cpu_workers,
        ).items()
    }
    executor_limits = _executor_specs(
        slot_mapping,
        cpu_workers=args.cpu_workers,
    )

    try:
        for run_id, error in sorted(metadata_errors.items()):
            run = all_runs_by_id[run_id]
            results[run_id] = {
                **_matrix_record_base(run),
                "status": "failed",
                "reason": "invalid_run_metadata",
                "error": error,
                "attempts": 0,
                "retries_used": 0,
                "exit_code": 0,
                "cuda_visible_devices": "",
            }
        persist_index()
        if metadata_errors:
            return 1
        pending_by_executor: dict[str, list[MatrixRun]] = {key: [] for key in executors}
        for run in runs:
            pending_by_executor[_executor_key_for(run, slot_mapping)].append(run)
        active_counts = {key: 0 for key in executors}
        future_to_run: dict[concurrent.futures.Future[dict[str, Any]], tuple[MatrixRun, str]] = {}

        def submit_available_runs() -> bool:
            submitted = False
            if args.fail_fast and stop_event.is_set():
                return submitted
            for executor_key, executor in executors.items():
                queue = pending_by_executor[executor_key]
                while queue and active_counts[executor_key] < executor_limits[executor_key]:
                    if args.fail_fast and stop_event.is_set():
                        return submitted
                    run = queue.pop(0)
                    future = executor.submit(execute, run)
                    future_to_run[future] = (run, executor_key)
                    active_counts[executor_key] += 1
                    submitted = True
            return submitted

        submit_available_runs()

        while future_to_run:
            done, _ = concurrent.futures.wait(
                list(future_to_run.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                run, executor_key = future_to_run.pop(future)
                active_counts[executor_key] = max(0, active_counts[executor_key] - 1)
                try:
                    record = future.result()
                except Exception as exc:
                    finished_at = time.time()
                    record = {
                        **_matrix_record_base(run, metadata=run_metadata_map.get(run.run_id)),
                        "status": "failed",
                        "reason": "runner_exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "attempts": 0,
                        "retries_used": 0,
                        "exit_code": 0,
                        "finished_at": finished_at,
                        "duration_seconds": 0.0,
                        "cuda_visible_devices": "",
                    }
                    with results_lock:
                        results[run.run_id] = record
                    persist_index()
                    if args.fail_fast:
                        stop_event.set()
                print(json.dumps(record, indent=2, sort_keys=True))
            submit_available_runs()

        if args.fail_fast and stop_event.is_set():
            for executor_key, queue in pending_by_executor.items():
                while queue:
                    run = queue.pop(0)
                    metadata = run_metadata_map.get(run.run_id, _blank_metadata())
                    record = {
                        **_matrix_record_base(run, metadata=metadata),
                        "status": "skipped",
                        "reason": "fail_fast_stop_requested",
                        "attempts": 0,
                        "retries_used": 0,
                        "exit_code": 0,
                        "cuda_visible_devices": "",
                    }
                    with results_lock:
                        results[run.run_id] = record
                    print(json.dumps(record, indent=2, sort_keys=True))
            persist_index()
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)
        _release_matrix_lock(lock_path, lock_owner)

    with results_lock:
        final_index = current_index()
    _write_index(index_path, final_index)
    failed = [item for item in final_index["runs"] if item["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
