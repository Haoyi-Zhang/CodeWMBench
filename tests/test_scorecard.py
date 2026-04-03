from __future__ import annotations

import pytest

from codewmbench.models import BenchmarkRow
from codewmbench.scorecard import scorecard_for_rows


def _score_row(*, example_id: str, model_label: str, source_group: str, task_category: str, human_detected: bool = False) -> BenchmarkRow:
    return BenchmarkRow(
        example_id=example_id,
        attack_name="comment_strip",
        task_id=example_id,
        dataset=source_group,
        language="python",
        task_category=task_category,
        reference_kind="canonical",
        method_origin="native",
        model_label=model_label,
        source_group=source_group,
        origin_type="public",
        difficulty="easy",
        watermark_scheme="kgw",
        clean_score=0.9,
        attacked_score=0.9,
        clean_detected=True,
        attacked_detected=True,
        quality_score=1.0,
        stealth_score=1.0,
        mutation_distance=0.0,
        watermark_retention=1.0,
        robustness_score=1.0,
        semantic_validation_available=True,
        semantic_preserving=True,
        metadata={
            "negative_controls": {
                "human_reference": {"available": True, "score": 0.8 if human_detected else 0.1, "detected": human_detected, "threshold": 0.5, "metadata": {}},
                "clean_generation": {"available": True, "score": 0.2, "detected": False, "threshold": 0.5, "metadata": {}},
            },
            "clean_functional_trials": [{"sample_index": 0, "compile_success": True, "passed": True, "error_kind": ""}],
            "clean_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "attacked_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "example_metadata": {"validation_supported": True, "category": task_category},
        },
    )


def test_scorecard_hits_perfect_score_for_perfect_rows() -> None:
    rows = [
        _score_row(example_id="e1", model_label="model-a", source_group="public_humaneval_plus", task_category="strings/parsing"),
        _score_row(example_id="e2", model_label="model-b", source_group="crafted_translation", task_category="graphs/search"),
    ]

    scorecard = scorecard_for_rows(rows)

    assert scorecard["detection_reliability"] == pytest.approx(1.0)
    assert scorecard["robustness"] == pytest.approx(1.0)
    assert scorecard["utility"] == pytest.approx(1.0)
    assert scorecard["stealth"] == pytest.approx(1.0)
    assert scorecard["generalization"] == pytest.approx(1.0)
    assert scorecard["CodeWMScore"] == pytest.approx(100.0)


def test_scorecard_gate_penalizes_high_negative_control_false_positive_rate() -> None:
    rows = [
        _score_row(example_id="e1", model_label="model-a", source_group="public_humaneval_plus", task_category="strings/parsing", human_detected=True),
        _score_row(example_id="e2", model_label="model-b", source_group="crafted_translation", task_category="graphs/search", human_detected=False),
    ]

    scorecard = scorecard_for_rows(rows)

    assert scorecard["negative_control_fpr"] == pytest.approx(0.25)
    assert scorecard["detection_reliability"] == pytest.approx(0.9)
    assert scorecard["gate"] == pytest.approx(0.75)
    assert scorecard["CodeWMScore"] == pytest.approx(73.116)


def test_scorecard_excludes_singleton_generalization_axes_from_average() -> None:
    rows = [
        _score_row(example_id="e1", model_label="model-a", source_group="public_humaneval_plus", task_category="strings/parsing"),
        _score_row(example_id="e2", model_label="model-a", source_group="crafted_translation", task_category="graphs/search"),
    ]

    scorecard = scorecard_for_rows(rows)

    assert scorecard["cross_model_stability"] is None
    assert scorecard["cross_source_stability"] == pytest.approx(1.0)
    assert scorecard["cross_task_stability"] is None
    assert scorecard["generalization"] == pytest.approx(1.0)
    assert scorecard["score_coverage"]["generalization_axes_used"] == ["cross_source"]


def test_scorecard_uses_executed_semantic_coverage_not_declared_only_support() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    row = BenchmarkRow(
        **{
            **row.as_dict(),
            "semantic_validation_available": False,
            "semantic_preserving": None,
            "metadata": {
                **dict(row.metadata),
                "example_metadata": {"validation_supported": True, "category": "strings/parsing"},
            },
        }
    )

    scorecard = scorecard_for_rows([row], include_generalization=False)

    assert scorecard["semantic_validation_rate"] == pytest.approx(0.0)
    assert scorecard["declared_semantic_validation_rate"] == pytest.approx(1.0)
    assert scorecard["robustness"] < 1.0
    assert scorecard["utility"] < 1.0


def test_scorecard_penalizes_missing_clean_generation_negatives_for_generation_time_rows() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    row = BenchmarkRow(
        **{
            **row.as_dict(),
            "evaluation_track": "generation_time",
            "metadata": {
                **dict(row.metadata),
                "provider_mode": "local_hf",
                "negative_controls": {
                    "human_reference": {"available": True, "score": 0.1, "detected": False, "threshold": 0.5, "metadata": {}},
                    "clean_generation": {"available": False, "score": 0.0, "detected": False, "threshold": 0.5, "metadata": {}},
                },
                "example_metadata": {
                    **dict(dict(row.metadata).get("example_metadata", {})),
                    "provider_generation_succeeded": True,
                },
            },
        }
    )

    scorecard = scorecard_for_rows([row])

    assert scorecard["negative_control_support_rate"] == pytest.approx(0.5)
    assert scorecard["gate"] == pytest.approx(0.5)
    assert scorecard["CodeWMScore"] < 50.0


def test_scorecard_does_not_award_detection_credit_without_available_negative_controls() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    row = BenchmarkRow(
        **{
            **row.as_dict(),
            "metadata": {
                **dict(row.metadata),
                "negative_controls": {
                    "human_reference": {"applicable": True, "available": False, "score": 0.0, "detected": False, "threshold": 0.5, "metadata": {}},
                    "clean_generation": {"applicable": True, "available": False, "score": 0.0, "detected": False, "threshold": 0.5, "metadata": {}},
                },
            },
        }
    )

    scorecard = scorecard_for_rows([row])

    assert scorecard["negative_control_support_rate"] == pytest.approx(0.0)
    assert scorecard["negative_control_fpr"] == pytest.approx(1.0)
    assert scorecard["detection_reliability"] == pytest.approx(0.0)
    assert scorecard["gate"] == pytest.approx(0.0)
    assert scorecard["CodeWMScore"] == pytest.approx(0.0)


def test_scorecard_does_not_penalize_clean_generation_when_it_is_not_applicable() -> None:
    row = _score_row(
        example_id="e1",
        model_label="reference_oracle",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    row = BenchmarkRow(
        **{
            **row.as_dict(),
            "evaluation_track": "reference_code",
            "metadata": {
                **dict(row.metadata),
                "negative_controls": {
                    "human_reference": {"applicable": True, "available": True, "score": 0.1, "detected": False, "threshold": 0.5, "metadata": {}},
                    "clean_generation": {"applicable": False, "available": False, "score": 0.0, "detected": False, "threshold": 0.5, "metadata": {}},
                },
            },
        }
    )

    scorecard = scorecard_for_rows([row], include_generalization=False)

    assert scorecard["negative_control_support_rate"] == pytest.approx(1.0)
    assert scorecard["negative_control_fpr"] == pytest.approx(0.0)
    assert scorecard["detection_reliability"] == pytest.approx(1.0)


def test_scorecard_can_balance_atomic_sources_without_row_count_domination() -> None:
    good = _score_row(
        example_id="good",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    bad_rows = []
    for index in range(6):
        degraded = _score_row(
            example_id=f"bad-{index}",
            model_label="model-a",
            source_group="crafted_translation",
            task_category="graphs/search",
            human_detected=True,
        )
        bad_rows.append(
            BenchmarkRow(
                **{
                    **degraded.as_dict(),
                    "attacked_detected": False,
                    "quality_score": 0.1,
                    "stealth_score": 0.1,
                    "watermark_retention": 0.0,
                    "metadata": {
                        **dict(degraded.metadata),
                        "attacked_validation": {
                            "available": True,
                            "passed": False,
                            "metadata": {"compile_success": False, "error_kind": "assertion"},
                        },
                    },
                }
            )
        )

    unbalanced = scorecard_for_rows([good, *bad_rows])
    balanced = scorecard_for_rows([good, *bad_rows], balance_by_source_group=True)

    assert balanced["score_coverage"]["aggregation_mode"] == "source_balanced"
    assert balanced["score_coverage"]["aggregated_source_groups"] == ["crafted_translation", "public_humaneval_plus"]
    assert balanced["score_coverage"]["source_balanced_sources"]["crafted_translation"]["slice_core"] >= 0.0
    assert balanced["score_coverage"]["source_balanced_sources"]["public_humaneval_plus"]["slice_core"] >= 0.0
    expected_slice_core = (
        balanced["score_coverage"]["source_balanced_sources"]["crafted_translation"]["slice_core"]
        + balanced["score_coverage"]["source_balanced_sources"]["public_humaneval_plus"]["slice_core"]
    ) / 2.0
    assert balanced["slice_core"] == pytest.approx(expected_slice_core, rel=1e-3)
    assert balanced["CodeWMScore"] > unbalanced["CodeWMScore"]
