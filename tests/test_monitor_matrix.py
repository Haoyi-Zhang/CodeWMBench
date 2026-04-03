from __future__ import annotations

import json
from pathlib import Path

from scripts import monitor_matrix


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def test_render_snapshot_shows_model_progress_active_runs_completed_models_and_gpu_state(tmp_path: Path) -> None:
    running_dir = tmp_path / "run_a"
    done_dir = tmp_path / "run_b"
    complete_model_dir = tmp_path / "run_c"
    _write_json(
        running_dir / "progress.json",
        {
            "status": "running",
            "stage": "attack_start",
            "example_index": 4,
            "total_examples": 10,
            "rows_completed": 12,
            "latest_example_id": "example-004",
            "latest_attack": "identifier_rename",
        },
    )
    _write_json(
        done_dir / "progress.json",
        {
            "status": "completed",
            "stage": "complete",
            "example_index": 10,
            "total_examples": 10,
            "rows_completed": 40,
            "latest_example_id": "example-010",
            "latest_attack": "budgeted_adaptive",
        },
    )
    _write_json(
        complete_model_dir / "progress.json",
        {
            "status": "completed",
            "stage": "complete",
            "example_index": 12,
            "total_examples": 12,
            "rows_completed": 48,
            "latest_example_id": "example-012",
            "latest_attack": "budgeted_adaptive",
        },
    )
    index_path = tmp_path / "matrix_index.json"
    _write_json(
        index_path,
        {
            "profile": "suite_all_models_methods",
            "run_count": 4,
            "completed_count": 2,
            "success_count": 2,
            "skipped_count": 0,
            "running_count": 1,
            "failed_count": 0,
            "pending_count": 1,
            "updated_at": 1000.0,
            "gpu_slot_mapping": {"default": [0, 1]},
            "runs": [
                {
                    "run_id": "suite_qwen25_14b_crafted_original_ewd_runtime",
                    "status": "running",
                    "effective_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                    "watermark_name": "ewd_runtime",
                    "benchmark_label": "Crafted Original",
                    "output_dir": str(running_dir),
                    "cuda_visible_devices": "0",
                    "started_at": 900.0,
                    "duration_seconds": 0.0,
                    "priority": 9,
                },
                {
                    "run_id": "suite_qwen25_14b_humaneval_plus_stone_runtime",
                    "status": "success",
                    "effective_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                    "watermark_name": "stone_runtime",
                    "benchmark_label": "HumanEval+",
                    "output_dir": str(done_dir),
                    "cuda_visible_devices": "1",
                    "started_at": 800.0,
                    "duration_seconds": 120.0,
                    "priority": 8,
                },
                {
                    "run_id": "suite_qwen25_7b_mbxp_kgw_runtime",
                    "status": "pending",
                    "effective_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "watermark_name": "kgw_runtime",
                    "benchmark_label": "MBXP-5lang (py/cpp/java slice)",
                    "output_dir": str(tmp_path / "run_d"),
                    "cuda_visible_devices": "",
                    "started_at": None,
                    "duration_seconds": 0.0,
                    "priority": 3,
                },
                {
                    "run_id": "suite_starcoder2_7b_mbpp_stone_runtime",
                    "status": "success",
                    "effective_model": "bigcode/starcoder2-7b",
                    "watermark_name": "stone_runtime",
                    "benchmark_label": "MBPP+",
                    "output_dir": str(complete_model_dir),
                    "cuda_visible_devices": "1",
                    "started_at": 760.0,
                    "duration_seconds": 90.0,
                    "priority": 1,
                },
            ],
        },
    )

    output = monitor_matrix.render_snapshot(
        index_path,
        gpu_rows=[
            {
                "index": 0,
                "name": "NVIDIA A800-SXM4-40GB",
                "utilization_gpu": 95,
                "memory_used": 32000,
                "memory_total": 40960,
            }
        ],
        now=1000.0,
    )

    assert "CodeWMBench Monitor  profile=suite_all_models_methods" in output
    assert "Qwen/Qwen2.5-Coder-14B-Instruct" in output
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in output
    assert "bigcode/starcoder2-7b" in output
    assert "progress=40.0%" in output
    assert "stage=attack_start" in output
    assert "identifier_rename" in output
    assert "Completed models" in output
    assert "gpu0 NVIDIA A800-SXM4-40GB util=95%" in output
    assert "eta=" in output


def test_render_snapshot_handles_missing_gpu_and_progress_files(tmp_path: Path) -> None:
    index_path = tmp_path / "matrix_index.json"
    _write_json(
        index_path,
        {
            "profile": "suite_canary_heavy",
            "run_count": 1,
            "completed_count": 0,
            "success_count": 0,
            "skipped_count": 0,
            "running_count": 1,
            "failed_count": 0,
            "pending_count": 0,
            "updated_at": 2000.0,
            "gpu_slot_mapping": {"default": [0]},
            "runs": [
                {
                    "run_id": "run_without_progress",
                    "status": "running",
                    "effective_model": "bigcode/starcoder2-7b",
                    "watermark_name": "sweet_runtime",
                    "benchmark_label": "MBPP+",
                    "output_dir": str(tmp_path / "missing_progress"),
                    "cuda_visible_devices": "0",
                    "started_at": 1900.0,
                    "duration_seconds": 0.0,
                    "priority": 1,
                }
            ],
        },
    )

    output = monitor_matrix.render_snapshot(index_path, gpu_rows=[], now=2000.0)

    assert "Running runs" in output
    assert "progress=0.0%" in output
    assert "nvidia-smi unavailable" in output
