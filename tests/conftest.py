from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codewmbench import crafted_benchmarks as _crafted_benchmarks
from codewmbench import crafted_templates as _crafted_templates
from codewmbench.models import BenchmarkExample, ExperimentConfig, WatermarkSpec


def _compat_solution_source(category: str, language: str, name: str) -> str:
    prefix = name.rsplit("_", 1)[0]
    builders = {
        "canonicalize_tokens": _crafted_templates.strings_source,
        "sum_after_marker": _crafted_templates.arrays_source,
        "dominant_key": _crafted_templates.maps_source,
        "count_balanced_pairs": _crafted_templates.parsing_source,
        "bit_balance_score": _crafted_templates.bit_source,
        "total_interval_coverage": _crafted_templates.interval_source,
        "shortest_grid_path": _crafted_templates.graph_source,
        "max_non_adjacent_sum": _crafted_templates.dp_source,
        "inventory_total": _crafted_templates.stateful_source,
        "normalize_query": _crafted_templates.api_source,
    }
    builder = builders.get(prefix)
    if builder is not None:
        return builder(language, name)
    return _crafted_templates.solution_source(category, language, name)


_crafted_benchmarks.solution_source = _compat_solution_source
_crafted_templates.solution_source = _compat_solution_source


@pytest.fixture()
def sample_example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="example-001",
        language="python",
        prompt="Return the factorial of n.",
        reference_solution="""
def factorial(n):
    if n < 2:
        return 1
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result
        """.strip(),
        reference_tests=("assert factorial(5) == 120",),
        execution_tests=("assert factorial(5) == 120", "assert factorial(0) == 1"),
    )


@pytest.fixture()
def sample_spec() -> WatermarkSpec:
    return WatermarkSpec(
        name="kgw",
        secret="anonymous",
        payload="wm",
        strength=1.0,
        parameters={"threshold": 0.5},
    )


@pytest.fixture()
def sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=11,
        corpus_size=3,
        language="python",
        watermark_name="kgw",
        watermark_secret="anonymous",
        watermark_payload="wm",
        watermark_strength=1.0,
        attacks=("comment_strip", "identifier_rename", "whitespace_normalize", "control_flow_flatten", "budgeted_adaptive"),
        attack_parameters={
            "budgeted_adaptive": {
                "budget": 3,
                "min_quality": 0.4,
                "candidate_order": ["comment_strip", "whitespace_normalize", "identifier_rename", "control_flow_flatten"],
            }
        },
    )
