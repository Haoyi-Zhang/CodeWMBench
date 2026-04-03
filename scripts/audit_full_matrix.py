from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _shared import dump_json, load_json
from _hf_readiness import HFModelRequirement, cache_entry_paths, smoke_load_local_hf_model, validate_local_hf_cache
from codewmbench.config import build_experiment_config, load_config, merge_config_source, validate_experiment_config
from codewmbench.models import BenchmarkExample, WatermarkSpec
from codewmbench.providers import summarize_provider_configuration
from codewmbench.watermarks.registry import available_watermarks, build_watermark_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-run audit for the CodeWMBench suite matrix.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/matrices/suite_all_models_methods.json"))
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runtime-smoke", action="store_true", help="Execute a one-example smoke run for runtime methods.")
    parser.add_argument("--strict-hf-cache", action="store_true", help="Require official root caches and validate shard integrity.")
    parser.add_argument("--model-load-smoke", action="store_true", help="Offline-load all required local_hf models and run a minimal generate.")
    parser.add_argument("--skip-provider-credentials", action="store_true")
    parser.add_argument("--skip-hf-access", action="store_true")
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return dict(load_json(manifest_path))


def _load_profile_runs(manifest: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    return [dict(item) for item in manifest.get("runs", []) if str(item.get("profile", profile)) == profile]


def _resolved_profile(manifest: dict[str, Any], requested_profile: str | None) -> str:
    profile = str(requested_profile or manifest.get("profile", "suite_all_models_methods")).strip()
    if not profile:
        return "suite_all_models_methods"
    return profile


def _resolved_run_raw(run_item: dict[str, Any]) -> dict[str, Any]:
    source = load_config(Path(str(run_item["config"])))
    overrides = dict(run_item.get("config_overrides", {})) if isinstance(run_item.get("config_overrides"), dict) else {}
    return merge_config_source(source, **overrides)


def _filter_issues(issues: list[str], *, skip_provider_credentials: bool) -> list[str]:
    if not skip_provider_credentials:
        return issues
    return [issue for issue in issues if "requires credential" not in issue]


def _example_for(language: str) -> BenchmarkExample:
    solution = {
        "python": "def add(a, b):\n    return a + b\n",
        "javascript": "function add(a, b) {\n  return a + b;\n}\n",
        "java": "class Add { int add(int a, int b) { return a + b; } }\n",
    }.get(language, "def add(a, b):\n    return a + b\n")
    return BenchmarkExample(
        example_id=f"audit_{language}",
        language=language,
        prompt="Write a function that adds two integers.",
        reference_solution=solution,
        execution_tests=("assert add(1, 2) == 3",) if language == "python" else (),
        metadata={"task_id": f"audit_{language}", "dataset": "audit"},
    )


def _smoke_project_native(method: str) -> dict[str, Any]:
    language = "python" if method != "structural_flow" else "python"
    example = _example_for(language)
    spec = WatermarkSpec(name=method, secret="audit", payload="wm", strength=0.55, parameters={"threshold": 0.5, "seed": 7})
    bundle = build_watermark_bundle(method)
    prepared = bundle.prepare_example(example, spec)
    watermarked = bundle.embed(prepared, spec)
    detection = bundle.detect(watermarked, spec, example_id=example.example_id)
    return {
        "method": method,
        "status": "ok" if detection.detected else "failed",
        "prepared_language": prepared.language,
        "watermarked_length": len(watermarked.source),
        "detected": bool(detection.detected),
        "score": float(detection.score),
    }


def _probe_hf_model(model_id: str, token: str, timeout: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(f"https://huggingface.co/{model_id}/resolve/main/config.json", method="HEAD")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"model": model_id, "accessible": True, "status": getattr(response, "status", 200), "reason": "ok"}
    except urllib.error.HTTPError as exc:
        return {"model": model_id, "accessible": False, "status": exc.code, "reason": str(exc.reason)}
    except urllib.error.URLError as exc:
        return {"model": model_id, "accessible": False, "status": None, "reason": str(exc.reason)}


def _hf_cache_entry(model_id: str, cache_dir: str) -> Path:
    root_entry, hub_entry = cache_entry_paths(model_id, cache_dir)
    if root_entry.exists():
        return root_entry
    if hub_entry.exists():
        return hub_entry
    return root_entry


def _merge_hf_model(
    bucket: dict[str, dict[str, Any]],
    *,
    model_name: str,
    cache_dir: str,
    local_files_only: bool,
    token_env: str,
    trust_remote_code: bool,
    device: str,
    dtype: str,
    usage: str,
    config_path: Path,
) -> None:
    entry = bucket.setdefault(
        model_name,
        {
            "model": model_name,
            "token_env": token_env,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "device": device,
            "dtype": dtype,
            "usage": set(),
            "config_paths": set(),
        },
    )
    entry["usage"].add(usage)
    entry["config_paths"].add(str(config_path))
    if cache_dir:
        entry["cache_dir"] = cache_dir
    entry["local_files_only"] = bool(entry["local_files_only"] or local_files_only)
    entry["trust_remote_code"] = bool(entry["trust_remote_code"] or trust_remote_code)
    if token_env:
        entry["token_env"] = token_env
    if device:
        entry["device"] = device
    if dtype:
        entry["dtype"] = dtype


def _requirement_from_model_config(item: dict[str, Any]) -> HFModelRequirement:
    return HFModelRequirement(
        model=str(item["model"]),
        cache_dir=str(item.get("cache_dir", "")),
        local_files_only=bool(item.get("local_files_only", False)),
        trust_remote_code=bool(item.get("trust_remote_code", False)),
        device=str(item.get("device", "cuda")),
        dtype=str(item.get("dtype", "float16")),
        token_env=str(item.get("token_env", "HF_ACCESS_TOKEN")),
        usage=tuple(sorted(str(entry) for entry in item.get("usage", set()))),
        config_paths=tuple(sorted(str(entry) for entry in item.get("config_paths", set()))),
    )


def _runtime_smoke(run_item: dict[str, Any]) -> dict[str, Any]:
    raw = json.loads(json.dumps(_resolved_run_raw(run_item)))
    raw.setdefault("project", {})
    raw["project"]["name"] = f"{raw['project'].get('name', 'audit')}-audit"
    benchmark = dict(raw.get("benchmark", {}))
    benchmark["limit"] = 1
    raw["benchmark"] = benchmark
    raw["attacks"] = {"include": ["comment_strip"]}
    audit_root = ROOT / "results" / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="codewmbench-audit-", dir=str(audit_root)))
    try:
        report_path = temp_dir / "report.json"
        payload = base64.b64encode(json.dumps(raw).encode("utf-8")).decode("ascii")
        child = (
            "import base64, json, os; "
            "from codewmbench.config import build_experiment_config; "
            "from codewmbench.pipeline import run_experiment; "
            "raw = json.loads(base64.b64decode(os.environ['CODEWMBENCH_AUDIT_RAW_CONFIG']).decode('utf-8')); "
            "config = build_experiment_config(raw, output_path=os.environ['CODEWMBENCH_AUDIT_REPORT_PATH']); "
            "run_experiment(config)"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                child,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "CODEWMBENCH_AUDIT_RAW_CONFIG": payload,
                "CODEWMBENCH_AUDIT_REPORT_PATH": str(report_path),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            },
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(details or f"run_experiment.py exited with status {completed.returncode}")
        payload = load_json(report_path)
        return {
            "method": str(raw.get("watermark", {}).get("scheme", "runtime")),
            "status": "ok",
            "row_count": len(payload.get("rows", [])),
            "report_path": str(report_path),
        }
    except Exception as exc:
        return {
            "method": str(raw.get("watermark", {}).get("scheme", "runtime")),
            "status": "failed",
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _smoke_methods_for_matrix(matrix_methods: set[str]) -> list[str]:
    ordered = [method for method in available_watermarks() if method in matrix_methods]
    extras = sorted(method for method in matrix_methods if method not in ordered)
    return ordered + extras


def _required_roster(values: list[Any] | None) -> list[str]:
    if not values:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _effective_model(config: Any) -> str:
    provider_model = str(config.provider_parameters.get("model", "")).strip()
    if provider_model:
        return provider_model
    return str(config.metadata.get("watermark", {}).get("model_name", "")).strip()


def main() -> int:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    profile = _resolved_profile(manifest, args.profile)
    output_path = args.output or Path(f"results/audits/{profile}_audit.json")
    run_items = _load_profile_runs(manifest, profile)
    configs: list[dict[str, Any]] = []
    matrix_methods: set[str] = set()
    provider_modes: set[str] = set()
    gpu_pools: set[str] = set()
    matrix_models: set[str] = set()
    benchmark_labels: set[str] = set()
    hf_models: dict[str, dict[str, Any]] = {}
    runtime_run_lookup: dict[str, dict[str, Any]] = {}
    slice_coverage: dict[tuple[str, str], set[str]] = {}
    issues: list[str] = []

    for item in run_items:
        config_path = Path(str(item["config"]))
        config = build_experiment_config(_resolved_run_raw(item))
        config_issues = _filter_issues(
            validate_experiment_config(config),
            skip_provider_credentials=args.skip_provider_credentials,
        )
        if config_issues:
            issues.extend(f"{config_path}: {issue}" for issue in config_issues)
        matrix_methods.add(config.watermark_name)
        provider_modes.add(config.provider_mode)
        effective_model = _effective_model(config)
        benchmark_label = str(config.corpus_parameters.get("dataset_label", "")).strip()
        gpu_pool = str(item.get("gpu_pool", "")).strip()
        if gpu_pool:
            gpu_pools.add(gpu_pool)
        if effective_model:
            matrix_models.add(effective_model)
        if benchmark_label:
            benchmark_labels.add(benchmark_label)
        if effective_model and benchmark_label:
            slice_coverage.setdefault((effective_model, benchmark_label), set()).add(config.watermark_name)
        provider_summary = summarize_provider_configuration(config.provider_mode, dict(config.provider_parameters))
        record = {
            "run_id": str(item["run_id"]),
            "config": str(config_path),
            "provider_mode": config.provider_mode,
            "watermark_name": config.watermark_name,
            "effective_model": effective_model,
            "benchmark_label": benchmark_label,
            "corpus_size": config.corpus_size,
            "provider_summary": provider_summary,
            "config_issues": config_issues,
        }
        if config.provider_mode == "local_hf":
            model_name = str(config.provider_parameters.get("model", "")).strip()
            token_env = str(config.provider_parameters.get("token_env", "HF_ACCESS_TOKEN")).strip()
            cache_dir = str(config.provider_parameters.get("cache_dir", "")).strip()
            local_files_only = bool(config.provider_parameters.get("local_files_only", False))
            if model_name:
                _merge_hf_model(
                    hf_models,
                    model_name=model_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    token_env=token_env,
                    trust_remote_code=bool(config.provider_parameters.get("trust_remote_code", False)),
                    device=str(config.provider_parameters.get("device", "cuda")),
                    dtype=str(config.provider_parameters.get("dtype", "float16")),
                    usage="local_hf",
                    config_path=config_path,
                )
        watermark_meta = dict(config.metadata.get("watermark", {})) if isinstance(config.metadata, dict) else {}
        runtime_model = str(watermark_meta.get("model_name", "")).strip()
        runtime_token_env = str(watermark_meta.get("token_env", "HF_ACCESS_TOKEN")).strip()
        runtime_cache_dir = str(watermark_meta.get("cache_dir", "")).strip()
        runtime_local_files_only = bool(watermark_meta.get("local_files_only", False))
        if runtime_model:
            _merge_hf_model(
                hf_models,
                model_name=runtime_model,
                cache_dir=runtime_cache_dir,
                local_files_only=runtime_local_files_only,
                token_env=runtime_token_env,
                trust_remote_code=bool(watermark_meta.get("trust_remote_code", False)),
                device=str(watermark_meta.get("device", "cuda")),
                dtype="float16",
                usage="runtime",
                config_path=config_path,
            )
        if config.watermark_name.endswith("_runtime"):
            runtime_run_lookup.setdefault(config.watermark_name, dict(item))
        configs.append(record)

    required_methods = [
        str(method).strip()
        for method in manifest.get("required_watermark_methods", list(available_watermarks()))
        if str(method).strip()
    ]
    required_provider_modes = [
        str(mode).strip()
        for mode in manifest.get("required_provider_modes", [])
        if str(mode).strip()
    ]
    required_gpu_pools = [
        str(pool).strip()
        for pool in manifest.get("required_gpu_pools", [])
        if str(pool).strip()
    ]
    required_model_roster = _required_roster(manifest.get("model_roster"))
    required_benchmark_roster = _required_roster(manifest.get("benchmark_roster"))

    missing_methods = sorted(set(required_methods) - matrix_methods)
    if missing_methods:
        issues.append(f"matrix is missing watermark methods: {missing_methods}")
    missing_provider_modes = sorted(set(required_provider_modes) - provider_modes)
    if missing_provider_modes:
        issues.append(f"matrix is missing provider modes: {missing_provider_modes}")
    missing_gpu_pools = sorted(set(required_gpu_pools) - gpu_pools)
    if missing_gpu_pools:
        issues.append(f"matrix is missing gpu pools: {missing_gpu_pools}")
    missing_model_roster = sorted(set(required_model_roster) - matrix_models)
    if missing_model_roster:
        issues.append(f"matrix is missing model_roster entries: {missing_model_roster}")
    missing_benchmark_roster = sorted(set(required_benchmark_roster) - benchmark_labels)
    if missing_benchmark_roster:
        issues.append(f"matrix is missing benchmark_roster entries: {missing_benchmark_roster}")

    missing_slice_methods: list[dict[str, Any]] = []
    if required_model_roster and required_benchmark_roster and required_methods:
        required_method_set = set(required_methods)
        for model_name in required_model_roster:
            for benchmark_label in required_benchmark_roster:
                present = slice_coverage.get((model_name, benchmark_label), set())
                missing = sorted(required_method_set - present)
                if missing:
                    missing_slice_methods.append(
                        {
                            "model": model_name,
                            "benchmark": benchmark_label,
                            "missing_methods": missing,
                        }
                    )
    if missing_slice_methods:
        issues.append(
            "matrix fairness coverage is incomplete for one or more (model, benchmark) slices"
        )

    method_smoke: list[dict[str, Any]] = []
    for method in _smoke_methods_for_matrix(matrix_methods):
        if method in {"stone_runtime", "sweet_runtime", "ewd_runtime", "kgw_runtime"}:
            if args.runtime_smoke:
                matching = runtime_run_lookup.get(method)
                if matching is None:
                    method_smoke.append({"method": method, "status": "failed", "error": "no_matrix_run"})
                else:
                    smoke = _runtime_smoke(matching)
                    method_smoke.append(smoke)
                    if smoke["status"] != "ok":
                        issues.append(f"runtime smoke failed for {method}: {smoke.get('error', 'unknown error')}")
            else:
                method_smoke.append({"method": method, "status": "skipped", "reason": "runtime_smoke_disabled"})
            continue
        smoke = _smoke_project_native(method)
        method_smoke.append(smoke)
        if smoke["status"] != "ok":
            issues.append(f"embed/detect smoke failed for {method}")

    required_hf_models = sorted(hf_models)
    hf_cache_validation: list[dict[str, Any]] = []
    if args.strict_hf_cache:
        for model_name in required_hf_models:
            requirement = _requirement_from_model_config(hf_models[model_name])
            result = validate_local_hf_cache(requirement, require_root_entry=True)
            result["usage"] = list(requirement.usage)
            result["config_paths"] = list(requirement.config_paths)
            hf_cache_validation.append(result)
            if result["status"] != "ok":
                issues.append(f"strict HF cache validation failed for {model_name}")

    hf_model_smoke: list[dict[str, Any]] = []
    if args.model_load_smoke:
        cache_status = {item["model"]: item for item in hf_cache_validation}
        for model_name in required_hf_models:
            requirement = _requirement_from_model_config(hf_models[model_name])
            if "local_hf" not in requirement.usage:
                continue
            if cache_status.get(model_name, {}).get("status") == "failed":
                smoke = {
                    "model": model_name,
                    "status": "skipped",
                    "issues": ["cache_validation_failed"],
                    "usage": list(requirement.usage),
                    "config_paths": list(requirement.config_paths),
                }
            else:
                smoke = smoke_load_local_hf_model(requirement)
                smoke["usage"] = list(requirement.usage)
                smoke["config_paths"] = list(requirement.config_paths)
                if smoke["status"] != "ok":
                    issues.append(f"offline model-load smoke failed for {model_name}: {smoke['issues'][0]}")
            hf_model_smoke.append(smoke)

    hf_access: list[dict[str, Any]] = []
    if not args.skip_hf_access:
        for model_name, access_config in sorted(hf_models.items()):
            token_env = str(access_config.get("token_env", "HF_ACCESS_TOKEN"))
            cache_dir = str(access_config.get("cache_dir", "")).strip()
            local_files_only = bool(access_config.get("local_files_only", False))
            if local_files_only and cache_dir:
                cache_entry = _hf_cache_entry(model_name, cache_dir)
                snapshots_dir = cache_entry / "snapshots"
                refs_dir = cache_entry / "refs"
                cached = cache_entry.exists() and (snapshots_dir.exists() or refs_dir.exists())
                result = {
                    "model": model_name,
                    "accessible": cached,
                    "status": "cache",
                    "reason": "local_cache" if cached else "missing_local_cache",
                }
            else:
                token = os.environ.get(token_env, "")
                result = _probe_hf_model(model_name, token)
            result["token_env"] = token_env
            result["cache_dir"] = cache_dir
            result["local_files_only"] = local_files_only
            hf_access.append(result)
            if not result["accessible"]:
                issues.append(f"Hugging Face access check failed for {model_name}: {result['status']} {result['reason']}")

    payload = {
        "profile": profile,
        "manifest": str(args.manifest),
        "config_count": len(configs),
        "method_count": len(required_methods),
        "required_methods": required_methods,
        "matrix_methods": sorted(matrix_methods),
        "missing_methods": missing_methods,
        "required_provider_modes": required_provider_modes,
        "provider_modes": sorted(provider_modes),
        "missing_provider_modes": missing_provider_modes,
        "required_gpu_pools": required_gpu_pools,
        "gpu_pools": sorted(gpu_pools),
        "missing_gpu_pools": missing_gpu_pools,
        "required_model_roster": required_model_roster,
        "matrix_models": sorted(matrix_models),
        "missing_model_roster": missing_model_roster,
        "required_benchmark_roster": required_benchmark_roster,
        "benchmark_labels": sorted(benchmark_labels),
        "missing_benchmark_roster": missing_benchmark_roster,
        "missing_slice_methods": missing_slice_methods,
        "configs": configs,
        "method_smoke": method_smoke,
        "required_hf_models": required_hf_models,
        "hf_cache_validation": hf_cache_validation,
        "hf_model_smoke": hf_model_smoke,
        "hf_access": hf_access,
        "issues": issues,
        "status": "clean" if not issues else "has_issues",
    }
    dump_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
