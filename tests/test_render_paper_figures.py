from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from codewmbench.models import BenchmarkRow, ExperimentConfig
from codewmbench.report import build_report
from codewmbench.suite import SUITE_AGGREGATE_SOURCE_GROUPS
from scripts import render_paper_figures


def _sample_row(
    *,
    example_id: str,
    attack_name: str,
    method: str,
    origin: str,
    model_label: str,
    source_group: str,
    task_category: str,
    language: str = "python",
    evaluation_track: str = "generation_time",
    clean_score: float = 0.92,
    attacked_score: float = 0.82,
    quality_score: float = 0.88,
    stealth_score: float = 0.9,
) -> BenchmarkRow:
    return BenchmarkRow(
        example_id=example_id,
        attack_name=attack_name,
        task_id=example_id,
        dataset=source_group,
        language=language,
        task_category=task_category,
        reference_kind="canonical",
        method_origin=origin,
        evaluation_track=evaluation_track,
        model_label=model_label,
        source_group=source_group,
        origin_type="public" if "public" in source_group else "crafted",
        difficulty="medium",
        watermark_scheme=method,
        clean_score=clean_score,
        attacked_score=attacked_score,
        clean_detected=True,
        attacked_detected=True,
        quality_score=quality_score,
        stealth_score=stealth_score,
        mutation_distance=1.0 - quality_score,
        watermark_retention=round(attacked_score / clean_score, 4),
        robustness_score=round(attacked_score * quality_score, 4),
        semantic_validation_available=True,
        semantic_preserving=True,
        metadata={
            "negative_controls": {
                "human_reference": {"available": True, "score": 0.08, "detected": False, "threshold": 0.5, "metadata": {}},
                "clean_generation": {"available": True, "score": 0.15, "detected": False, "threshold": 0.5, "metadata": {}},
            },
            "clean_functional_trials": [
                {"sample_index": 0, "compile_success": True, "passed": True, "error_kind": ""},
            ],
            "clean_functional_summary": {"compile_success_rate": 1.0, "test_pass_rate": 1.0},
            "clean_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "attacked_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "attack_metadata": {
                "budget_curve": [
                    {"budget": 0, "detector_score": clean_score, "quality_score": 1.0, "semantic_preserving": True},
                    {"budget": 1, "detector_score": attacked_score, "quality_score": quality_score, "semantic_preserving": True},
                ]
            },
            "example_metadata": {
                "validation_supported": True,
                "category": task_category,
                "source_group": source_group,
                "difficulty": "medium",
                "reference_kind": "canonical",
            },
        },
    )


def _write_report(path: Path, *, method: str, origin: str, model_label: str) -> None:
    evaluation_track = "reference_code" if model_label == "reference_oracle" else "generation_time"
    rows = [
        _sample_row(
            example_id=f"{method}-1",
            attack_name="budgeted_adaptive",
            method=method,
            origin=origin,
            model_label=model_label,
            source_group="public_humaneval_plus",
            task_category="strings/parsing",
            evaluation_track=evaluation_track,
            quality_score=0.9,
            stealth_score=0.92,
        ),
        _sample_row(
            example_id=f"{method}-2",
            attack_name="identifier_rename",
            method=method,
            origin=origin,
            model_label=model_label,
            source_group="crafted_translation",
            task_category="graphs/search",
            evaluation_track=evaluation_track,
            quality_score=0.82,
            stealth_score=0.84,
        ),
    ]
    report = build_report(ExperimentConfig(provider_mode="local_hf", watermark_name=method), rows, benchmark_manifest={})
    path.write_text(report.to_json(), encoding="utf-8")


def _write_suite_report(path: Path, *, method: str, origin: str, model_label: str) -> None:
    evaluation_track = "reference_code" if model_label == "reference_oracle" else "generation_time"
    rows = [
        _sample_row(
            example_id=f"{method}-{index}",
            attack_name="budgeted_adaptive" if index % 2 == 0 else "identifier_rename",
            method=method,
            origin=origin,
            model_label=model_label,
            source_group=source_group,
            task_category="strings/parsing" if "human" in source_group else "graphs/search",
            evaluation_track=evaluation_track,
            quality_score=0.88 - (index * 0.01),
            stealth_score=0.9 - (index * 0.01),
        )
        for index, source_group in enumerate(SUITE_AGGREGATE_SOURCE_GROUPS, start=1)
    ]
    report = build_report(ExperimentConfig(provider_mode="local_hf", watermark_name=method), rows, benchmark_manifest={})
    path.write_text(report.to_json(), encoding="utf-8")


def test_configure_matplotlib_uses_publication_safe_times_typography() -> None:
    _, plt = render_paper_figures.configure_matplotlib()
    assert plt.rcParams["font.family"][0] == "serif"
    assert plt.rcParams["font.serif"][0] == "Times New Roman"
    assert float(plt.rcParams["font.size"]) >= 11
    assert float(plt.rcParams["axes.titlesize"]) >= 12.5
    assert float(plt.rcParams["axes.labelsize"]) >= 10.5
    assert float(plt.rcParams["legend.fontsize"]) >= 9
    assert int(plt.rcParams["pdf.fonttype"]) == 42
    assert int(plt.rcParams["ps.fonttype"]) == 42


def test_presentation_row_normalizes_dataset_labels_and_stringified_lists() -> None:
    row = {
        "dataset": "HumanEval-X",
        "datasets": "['CraftedOriginal', 'HumanEval-X', 'MBXP 5-language subset']",
    }
    normalized = render_paper_figures._presentation_row(row)
    assert normalized["dataset"] == "HumanEval-X (py/cpp/java slice)"
    assert normalized["datasets"] == "['Crafted Original', 'HumanEval-X (py/cpp/java slice)', 'MBXP-5lang (py/cpp/java slice)']"


def test_rendered_pdf_avoids_type3_fonts(tmp_path: Path) -> None:
    _, plt = render_paper_figures.configure_matplotlib()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("Font Smoke")
    output = tmp_path / "font_smoke.pdf"
    fig.savefig(output)
    payload = output.read_bytes()
    assert b"/Subtype /Type3" not in payload


def test_configure_matplotlib_requires_times_new_roman_when_requested(monkeypatch) -> None:
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [SimpleNamespace(name="DejaVu Serif")])

    with pytest.raises(RuntimeError, match="Times New Roman is not available"):
        render_paper_figures.configure_matplotlib(require_times_new_roman=True)


def test_render_figures_writes_core_outputs_and_leaderboards(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"language_counts": {"python": 120, "cpp": 120, "java": 120, "javascript": 120, "go": 120}}),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    _write_report(report_path, method="kgw", origin="native", model_label="Qwen/Qwen2.5-Coder-7B-Instruct")

    baseline_path = tmp_path / "baseline_eval.json"
    baseline_path.write_text(
        json.dumps(
            {
                "baseline_family": "runtime_official",
                "watermark_schemes": ["stone_runtime"],
                "datasets": ["public_humaneval_plus"],
                "source_groups": ["public_humaneval_plus"],
                "model_labels": ["bigcode/starcoder2-7b"],
                "evaluation_tracks": ["generation_time"],
                "clean_reference_vs_watermarked_auroc": 0.93,
                "watermarked_pass_rate": 0.9,
                "average_perplexity_watermarked": 12.4,
                "stem_clean_reference": 0.82,
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "figures"
    outputs = render_paper_figures.render_figures(
        manifest_path=manifest_path,
        report_paths=[report_path],
        baseline_eval_paths=[baseline_path],
        output_dir=output_dir,
        prefix="tosemcompact",
    )

    expected = {
        output_dir / "tosemcompact_language_coverage.pdf",
        output_dir / "tosemcompact_language_coverage.png",
        output_dir / "tosemcompact_language_coverage.json",
        output_dir / "tosemcompact_language_coverage.csv",
        output_dir / "tosemcompact_attack_robustness.pdf",
        output_dir / "tosemcompact_attack_robustness.png",
        output_dir / "tosemcompact_attack_robustness.json",
        output_dir / "tosemcompact_attack_robustness.csv",
        output_dir / "tosemcompact_functional_summary.pdf",
        output_dir / "tosemcompact_functional_summary.png",
        output_dir / "tosemcompact_functional_summary.json",
        output_dir / "tosemcompact_functional_summary.csv",
        output_dir / "tosemcompact_baseline_comparison.pdf",
        output_dir / "tosemcompact_baseline_comparison.png",
        output_dir / "tosemcompact_baseline_comparison.json",
        output_dir / "tosemcompact_baseline_comparison.csv",
        output_dir / "tosemcompact_method_model_leaderboard.json",
        output_dir / "tosemcompact_method_model_leaderboard.csv",
        output_dir / "tosemcompact_method_master_leaderboard.json",
        output_dir / "tosemcompact_method_master_leaderboard.csv",
        output_dir / "tosemcompact_upstream_only_leaderboard.json",
        output_dir / "tosemcompact_upstream_only_leaderboard.csv",
    }
    assert set(outputs) == expected
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)


def test_render_figures_suite_all_writes_colm_score_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"language_counts": {"python": 100, "cpp": 90, "java": 80, "javascript": 70, "go": 60}}),
        encoding="utf-8",
    )
    report_path = tmp_path / "upstream_report.json"
    _write_suite_report(report_path, method="stone_runtime", origin="upstream", model_label="bigcode/starcoder2-7b")
    generation_peer_report = tmp_path / "peer_report.json"
    _write_suite_report(generation_peer_report, method="sweet_runtime", origin="upstream", model_label="bigcode/starcoder2-7b")
    local_hf_report = tmp_path / "native_report.json"
    _write_suite_report(local_hf_report, method="comment", origin="native", model_label="reference_oracle")
    second_model_report = tmp_path / "second_model_report.json"
    _write_suite_report(second_model_report, method="ewd_runtime", origin="upstream", model_label="Qwen/Qwen2.5-Coder-14B-Instruct")

    baseline_eval = tmp_path / "baseline_eval.json"
    baseline_eval.write_text(
        json.dumps(
            {
                "baseline_family": "runtime_official",
                "watermark_schemes": ["stone_runtime"],
                "datasets": ["public_humaneval_plus"],
                "source_groups": ["public_humaneval_plus"],
                "model_labels": ["bigcode/starcoder2-7b"],
                "evaluation_tracks": ["generation_time"],
                "clean_reference_vs_watermarked_auroc": 0.93,
                "watermarked_pass_rate": 0.9,
                "average_perplexity_watermarked": 12.0,
                "stem_clean_reference": 0.81,
            }
        ),
        encoding="utf-8",
    )
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {"report_path": str(report_path), "baseline_eval_path": str(baseline_eval)},
                    {"report_path": str(generation_peer_report), "baseline_eval_path": ""},
                    {"report_path": str(local_hf_report), "baseline_eval_path": ""},
                    {"report_path": str(second_model_report), "baseline_eval_path": ""},
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "suite_all"
    outputs = render_paper_figures.render_figures(
        manifest_path=[manifest_path],
        report_paths=[],
        baseline_eval_paths=[],
        matrix_index_path=matrix_index,
        output_dir=output_dir,
        prefix="tosemcompact",
        suite="all",
        anchor_report_path=report_path,
        allow_mixed_tracks=True,
        include_reference_artifacts=True,
    )

    assert any(path.name == "tosemcompact_semantic_attack_robustness.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_reference_kind_comparison.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_budget_curve.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_by_language_metrics.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_runtime_baseline_ppl_stem.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_overall_leaderboard.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_public_only_overall_leaderboard.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_score_decomposition.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_generalization_breakdown.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_quality_vs_robustness.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_detection_vs_utility.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_per_source_breakdown.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_per_model_breakdown.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_per_language_breakdown.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_attack_breakdown.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_source_language_coverage.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_method_stability_heatmap.pdf" for path in outputs)
    assert any(path.name == "tosemcompact_reference_code_method_master_leaderboard.json" for path in outputs)
    assert any(path.name == "tosemcompact_public_only_method_master_leaderboard.json" for path in outputs)
    assert not any(path.name.endswith("_model_comparison.pdf") for path in outputs)
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    functional_payload = json.loads((output_dir / "tosemcompact_functional_summary.json").read_text(encoding="utf-8"))
    assert all(row["aggregation_mode"] == "suite_source_aggregate" for row in functional_payload)
    assert set(functional_payload[0]["source_groups"]) == set(SUITE_AGGREGATE_SOURCE_GROUPS)


def test_render_figures_suite_all_allows_optional_anchor_for_aggregate_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"language_counts": {"python": 100, "cpp": 90, "java": 80}}), encoding="utf-8")
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_suite_report(report_a, method="stone_runtime", origin="upstream", model_label="bigcode/starcoder2-7b")
    _write_suite_report(report_b, method="sweet_runtime", origin="upstream", model_label="bigcode/starcoder2-7b")

    outputs = render_paper_figures.render_figures(
        manifest_path=[manifest_path],
        report_paths=[report_a, report_b],
        baseline_eval_paths=[],
        output_dir=tmp_path / "suite_figures",
        prefix="suite",
        suite="all",
    )

    assert any(path.name == "suite_overall_leaderboard.pdf" for path in outputs)
    assert any(path.name == "suite_functional_summary.pdf" for path in outputs)
    assert not any(path.name == "suite_attack_robustness.pdf" for path in outputs)


def test_render_figures_suite_all_fails_when_atomic_source_coverage_is_incomplete(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"language_counts": {"python": 100}}), encoding="utf-8")
    report_path = tmp_path / "report.json"
    _write_suite_report(report_path, method="stone_runtime", origin="upstream", model_label="bigcode/starcoder2-7b")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["rows"] = payload["rows"][:-1]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lost atomic source groups"):
        render_paper_figures.render_figures(
            manifest_path=[manifest_path],
            report_paths=[report_path],
            baseline_eval_paths=[],
            output_dir=tmp_path / "figures",
            prefix="tosemcompact",
            suite="all",
        )


def test_render_figures_suite_all_fails_when_expected_model_roster_is_missing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"language_counts": {"python": 100, "cpp": 90, "java": 80}}), encoding="utf-8")
    report_path = tmp_path / "report.json"
    _write_suite_report(report_path, method="stone_runtime", origin="upstream", model_label="bigcode/starcoder2-7b")
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "resource": "gpu",
                        "status": "success",
                        "report_path": str(report_path),
                        "effective_model": "bigcode/starcoder2-7b",
                    },
                    {
                        "resource": "gpu",
                        "status": "success",
                        "report_path": str(tmp_path / "missing_report.json"),
                        "effective_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing expected models"):
        render_paper_figures.render_figures(
            manifest_path=[manifest_path],
            report_paths=[],
            baseline_eval_paths=[],
            matrix_index_path=matrix_index,
            output_dir=tmp_path / "figures",
            prefix="suite",
            suite="all",
            anchor_report_path=report_path,
            allow_mixed_tracks=True,
        )


def test_render_figures_rejects_mixed_track_inputs_by_default(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"language_counts": {"python": 100}}), encoding="utf-8")

    generation_report = tmp_path / "generation_report.json"
    _write_report(generation_report, method="stone_runtime", origin="upstream", model_label="bigcode/starcoder2-7b")

    reference_report = tmp_path / "reference_report.json"
    _write_report(reference_report, method="comment", origin="native", model_label="reference_oracle")

    with pytest.raises(ValueError, match="single explicit track"):
        render_paper_figures.render_figures(
            manifest_path=[manifest_path],
            report_paths=[generation_report, reference_report],
            baseline_eval_paths=[],
            output_dir=tmp_path / "figures",
            prefix="tosemcompact",
            suite="all",
        )


def test_render_figures_rejects_reports_without_explicit_track(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"language_counts": {"python": 100}}), encoding="utf-8")

    row = _sample_row(
        example_id="legacy-1",
        attack_name="comment_strip",
        method="stone_runtime",
        origin="upstream",
        model_label="bigcode/starcoder2-7b",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
        evaluation_track="",
    )
    report = build_report(ExperimentConfig(provider_mode="local_hf", watermark_name="stone_runtime"), [row], benchmark_manifest={})
    report_path = tmp_path / "legacy_report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one evaluation track each"):
        render_paper_figures.render_figures(
            manifest_path=[manifest_path],
            report_paths=[report_path],
            baseline_eval_paths=[],
            output_dir=tmp_path / "figures",
            prefix="tosemcompact",
            suite="all",
        )

