from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from codewmbench.suite import (
    ACTIVE_SUITE_LIMITS,
    OFFICIAL_RUNTIME_BASELINES,
    OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
    SUITE_ATOMIC_SOURCE_ORDER,
    SUITE_MODEL_ROSTER,
)


def _manifest(path_name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs" / "matrices" / path_name).read_text(encoding="utf-8"))


def _normalize_language(row: dict) -> str:
    return str(row.get("language", "")).strip().lower()


def test_suite_full_manifest_matches_models_methods_and_atomic_sources() -> None:
    manifest = _manifest("suite_all_models_methods.json")

    assert manifest["profile"] == "suite_all_models_methods"
    assert manifest["schema_version"] == 1
    assert manifest["model_roster"] == list(SUITE_MODEL_ROSTER)
    assert manifest["method_roster"] == list(OFFICIAL_RUNTIME_BASELINES)
    assert manifest["atomic_benchmark_sources"] == list(SUITE_ATOMIC_SOURCE_ORDER)
    assert len(manifest["runs"]) == len(SUITE_MODEL_ROSTER) * len(OFFICIAL_RUNTIME_BASELINES) * len(SUITE_ATOMIC_SOURCE_ORDER)
    assert all(run["resource"] == "gpu" for run in manifest["runs"])
    assert all(run["gpu_pool"] == "runtime" for run in manifest["runs"])
    assert all(run["baseline_eval"] is True for run in manifest["runs"])


def test_suite_full_manifest_uses_complete_public_sources_and_balanced_crafted_subsets() -> None:
    manifest = _manifest("suite_all_models_methods.json")

    expected_limits = {
        "HumanEval+": 164,
        "MBPP+": ACTIVE_SUITE_LIMITS["mbpp_plus"],
        "HumanEval-X (py/cpp/java slice)": ACTIVE_SUITE_LIMITS["humaneval_x"],
        "MBXP-5lang (py/cpp/java slice)": ACTIVE_SUITE_LIMITS["mbxp_5lang"],
        "Crafted Original": ACTIVE_SUITE_LIMITS["crafted_original"],
        "Crafted Translation": ACTIVE_SUITE_LIMITS["crafted_translation"],
        "Crafted Stress": ACTIVE_SUITE_LIMITS["crafted_stress"],
    }
    for run in manifest["runs"]:
        benchmark = dict(run["config_overrides"]["benchmark"])
        dataset_label = str(benchmark["dataset_label"])
        assert int(benchmark["limit"]) == expected_limits[dataset_label]


def test_suite_stage_a_manifest_uses_heavy_model_across_atomic_sources() -> None:
    manifest = _manifest("suite_canary_heavy.json")

    assert manifest["profile"] == "suite_canary_heavy"
    assert len(manifest["model_roster"]) == 1
    assert manifest["method_roster"] == list(OFFICIAL_RUNTIME_BASELINES)
    assert manifest["atomic_benchmark_sources"] == list(SUITE_ATOMIC_SOURCE_ORDER)
    assert len(manifest["runs"]) == len(OFFICIAL_RUNTIME_BASELINES) * len(SUITE_ATOMIC_SOURCE_ORDER)
    for run in manifest["runs"]:
        benchmark = dict(run["config_overrides"]["benchmark"])
        dataset_label = str(benchmark["dataset_label"])
        if dataset_label in {"HumanEval+", "MBPP+"}:
            assert int(benchmark["limit"]) == 12
        else:
            assert int(benchmark["limit"]) == 15
        if dataset_label in {"HumanEval-X (py/cpp/java slice)", "MBXP-5lang (py/cpp/java slice)", "Crafted Original", "Crafted Translation", "Crafted Stress"}:
            assert set(benchmark["languages"]) == set(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES)
        assert int(run["baseline_eval_sample_limit"]) == 16


def test_suite_stage_b_manifest_smokes_remaining_models_on_python_sources() -> None:
    manifest = _manifest("model_invocation_smoke.json")

    assert manifest["profile"] == "model_invocation_smoke"
    assert len(manifest["model_roster"]) == len(SUITE_MODEL_ROSTER) - 1
    assert manifest["benchmark_roster"] == ["HumanEval+", "MBPP+"]
    assert manifest["atomic_benchmark_sources"] == ["humaneval_plus", "mbpp_plus"]
    assert len(manifest["runs"]) == (len(SUITE_MODEL_ROSTER) - 1) * len(OFFICIAL_RUNTIME_BASELINES) * 2
    for run in manifest["runs"]:
        benchmark = dict(run["config_overrides"]["benchmark"])
        assert int(benchmark["limit"]) == 2
        assert int(run["baseline_eval_sample_limit"]) == 4


def test_compact_prepared_sources_match_tosem_compact_sizes_and_common_support_languages() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "data/compact/collections/suite_mbpp_plus_compact.normalized.jsonl": (240, {"python"}),
        "data/compact/collections/suite_humanevalx_compact.normalized.jsonl": (120, {"python", "cpp", "java"}),
        "data/compact/collections/suite_mbxp_compact.normalized.jsonl": (120, {"python", "cpp", "java"}),
        "data/compact/collections/suite_crafted_original_compact.normalized.jsonl": (120, {"python", "cpp", "java"}),
        "data/compact/collections/suite_crafted_translation_compact.normalized.jsonl": (120, {"python", "cpp", "java"}),
        "data/compact/collections/suite_crafted_stress_compact.normalized.jsonl": (120, {"python", "cpp", "java"}),
    }
    for relative_path, (expected_count, expected_languages) in expected.items():
        path = root / relative_path
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == expected_count
        languages = {_normalize_language(row) for row in rows}
        assert languages == expected_languages
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert int(manifest["record_count"]) == expected_count
        if len(expected_languages) > 1:
            language_counts = Counter(_normalize_language(row) for row in rows)
            assert set(language_counts.values()) == {expected_count // len(expected_languages)}


def test_crafted_compact_sources_keep_balanced_family_slices() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "data/compact/collections/suite_crafted_original_compact.normalized.jsonl",
        "data/compact/collections/suite_crafted_translation_compact.normalized.jsonl",
        "data/compact/collections/suite_crafted_stress_compact.normalized.jsonl",
    ):
        path = root / relative_path
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert int(manifest["family_count"]) == 40
        assert manifest["suite_selection_policy"]["type"] == "category_balanced_multilingual_compact_slice"


def test_suite_manifests_align_sampling_seed_per_model_source_within_stage() -> None:
    for manifest_name in ("suite_all_models_methods.json", "suite_canary_heavy.json", "model_invocation_smoke.json"):
        manifest = _manifest(manifest_name)
        seeds_by_slice: dict[tuple[str, str], set[int]] = {}
        for run in manifest["runs"]:
            project = dict(run["config_overrides"]["project"])
            benchmark = dict(run["config_overrides"]["benchmark"])
            watermark = dict(run["config_overrides"]["watermark"])
            model_name = str(watermark["model_name"])
            source_group = str(benchmark["source_group"])
            key = (model_name, source_group)
            seeds_by_slice.setdefault(key, set()).add(int(project["seed"]))
        assert all(len(seeds) == 1 for seeds in seeds_by_slice.values())


def test_suite_full_manifest_drops_stale_collection_sources_from_compact_overrides() -> None:
    manifest = _manifest("suite_all_models_methods.json")

    compact_runs = [
        run
        for run in manifest["runs"]
        if "compact" in str(run["config_overrides"]["benchmark"].get("prepared_output", ""))
    ]

    assert compact_runs
    assert all("collection_sources" not in dict(run["config_overrides"]["benchmark"]) for run in compact_runs)


def test_suite_full_manifest_prioritizes_heavier_model_method_source_combinations() -> None:
    manifest = _manifest("suite_all_models_methods.json")
    by_run_id = {run["run_id"]: run for run in manifest["runs"]}

    qwen14_ewd_crafted_original = by_run_id["suite_qwen25_14b_crafted_original_ewd_runtime"]
    qwen14_stone_humaneval = by_run_id["suite_qwen25_14b_heplus_stone_runtime"]
    starcoder_stone_humaneval = by_run_id["suite_starcoder2_7b_heplus_stone_runtime"]

    assert int(qwen14_ewd_crafted_original["priority"]) > int(qwen14_stone_humaneval["priority"])
    assert int(qwen14_stone_humaneval["priority"]) > int(starcoder_stone_humaneval["priority"])
