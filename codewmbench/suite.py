from __future__ import annotations

from dataclasses import dataclass


OFFICIAL_BASELINE_ROSTER: tuple[str, ...] = (
    "stone_runtime",
    "sweet_runtime",
    "ewd_runtime",
    "kgw_runtime",
)
OFFICIAL_RUNTIME_BASELINES: tuple[str, ...] = OFFICIAL_BASELINE_ROSTER
OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES: tuple[str, ...] = ("python", "cpp", "java")

PAPER_MODEL_ROSTER: tuple[str, ...] = (
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "bigcode/starcoder2-7b",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
)
SUITE_MODEL_ROSTER: tuple[str, ...] = PAPER_MODEL_ROSTER

# Active TOSEM-compact sizing policy:
# - keep the benchmark inventory intact
# - keep HumanEval+ complete as the public Python anchor
# - reduce the remaining aggregate sources to balanced, deterministic compact slices
ACTIVE_SUITE_LIMITS: dict[str, int] = {
    "humaneval_plus": 164,
    "mbpp_plus": 240,
    "humaneval_x": 120,
    "mbxp_5lang": 120,
    "crafted_original": 120,
    "crafted_translation": 120,
    "crafted_stress": 120,
}

HEAVY_PRECHECK_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
SUITE_MODEL_SLUGS: dict[str, str] = {
    "Qwen/Qwen2.5-Coder-14B-Instruct": "qwen25_14b",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen25_7b",
    "bigcode/starcoder2-7b": "starcoder2_7b",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "deepseek_coder_6p7b",
}


@dataclass(frozen=True, slots=True)
class SuiteSourceSpec:
    slug: str
    dataset_label: str
    source_group: str
    prepared_benchmark: str
    prepared_output: str
    validation_scope: str
    languages: tuple[str, ...]
    base_template: str
    collection_name: str = ""
    public_source: str = ""
    include_reference_kinds: tuple[str, ...] = ("canonical",)
    full_limit: int = 0
    stage_a_limit: int = 0
    stage_b_limit: int = 0
    aggregate_score: bool = True


SUITE_INVENTORY_SOURCES: tuple[SuiteSourceSpec, ...] = (
    SuiteSourceSpec(
        slug="human_eval",
        dataset_label="HumanEval",
        source_group="public_human_eval",
        prepared_benchmark="data/public/human_eval/normalized.jsonl",
        prepared_output="data/public/human_eval/normalized.jsonl",
        validation_scope="python_first",
        languages=("python",),
        base_template="humaneval_plus",
        public_source="human_eval",
        full_limit=164,
        stage_a_limit=12,
        stage_b_limit=2,
        aggregate_score=False,
    ),
    SuiteSourceSpec(
        slug="humaneval_plus",
        dataset_label="HumanEval+",
        source_group="public_humaneval_plus",
        prepared_benchmark="data/public/humaneval_plus/normalized.jsonl",
        prepared_output="data/public/humaneval_plus/normalized.jsonl",
        validation_scope="python_first",
        languages=("python",),
        base_template="humaneval_plus",
        public_source="humaneval_plus",
        full_limit=164,
        stage_a_limit=12,
        stage_b_limit=2,
    ),
    SuiteSourceSpec(
        slug="mbpp_plus",
        dataset_label="MBPP+",
        source_group="public_mbpp_plus",
        prepared_benchmark="data/compact/collections/suite_mbpp_plus_compact.normalized.jsonl",
        prepared_output="data/compact/collections/suite_mbpp_plus_compact.normalized.jsonl",
        validation_scope="python_first",
        languages=("python",),
        base_template="mbpp_plus",
        public_source="mbpp_plus",
        full_limit=ACTIVE_SUITE_LIMITS["mbpp_plus"],
        stage_a_limit=12,
        stage_b_limit=2,
    ),
    SuiteSourceSpec(
        slug="humaneval_x",
        dataset_label="HumanEval-X (py/cpp/java slice)",
        source_group="public_humaneval_x",
        prepared_benchmark="data/compact/collections/suite_humanevalx_compact.normalized.jsonl",
        prepared_output="data/compact/collections/suite_humanevalx_compact.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="suite_humanevalx_compact",
        full_limit=ACTIVE_SUITE_LIMITS["humaneval_x"],
        stage_a_limit=15,
    ),
    SuiteSourceSpec(
        slug="mbxp_5lang",
        dataset_label="MBXP-5lang (py/cpp/java slice)",
        source_group="public_mbxp_5lang",
        prepared_benchmark="data/compact/collections/suite_mbxp_compact.normalized.jsonl",
        prepared_output="data/compact/collections/suite_mbxp_compact.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="suite_mbxp_compact",
        full_limit=ACTIVE_SUITE_LIMITS["mbxp_5lang"],
        stage_a_limit=15,
    ),
    SuiteSourceSpec(
        slug="crafted_original",
        dataset_label="Crafted Original",
        source_group="crafted_original",
        prepared_benchmark="data/compact/collections/suite_crafted_original_compact.normalized.jsonl",
        prepared_output="data/compact/collections/suite_crafted_original_compact.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="suite_crafted_original_compact",
        include_reference_kinds=(),
        full_limit=ACTIVE_SUITE_LIMITS["crafted_original"],
        stage_a_limit=15,
    ),
    SuiteSourceSpec(
        slug="crafted_translation",
        dataset_label="Crafted Translation",
        source_group="crafted_translation",
        prepared_benchmark="data/compact/collections/suite_crafted_translation_compact.normalized.jsonl",
        prepared_output="data/compact/collections/suite_crafted_translation_compact.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="suite_crafted_translation_compact",
        include_reference_kinds=(),
        full_limit=ACTIVE_SUITE_LIMITS["crafted_translation"],
        stage_a_limit=15,
    ),
    SuiteSourceSpec(
        slug="crafted_stress",
        dataset_label="Crafted Stress",
        source_group="crafted_stress",
        prepared_benchmark="data/compact/collections/suite_crafted_stress_compact.normalized.jsonl",
        prepared_output="data/compact/collections/suite_crafted_stress_compact.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="suite_crafted_stress_compact",
        full_limit=ACTIVE_SUITE_LIMITS["crafted_stress"],
        stage_a_limit=15,
    ),
)

SUITE_AGGREGATE_SOURCES: tuple[SuiteSourceSpec, ...] = tuple(
    source for source in SUITE_INVENTORY_SOURCES if source.aggregate_score
)
SUITE_ATOMIC_SOURCE_ORDER: tuple[str, ...] = tuple(source.slug for source in SUITE_AGGREGATE_SOURCES)
SUITE_ATOMIC_SOURCE_LABELS: dict[str, str] = {
    source.slug: source.dataset_label for source in SUITE_AGGREGATE_SOURCES
}

SUITE_INVENTORY_SOURCE_GROUPS: tuple[str, ...] = tuple(source.source_group for source in SUITE_INVENTORY_SOURCES)
SUITE_AGGREGATE_SOURCE_GROUPS: tuple[str, ...] = tuple(source.source_group for source in SUITE_AGGREGATE_SOURCES)
SUITE_INVENTORY_DATASETS: tuple[str, ...] = tuple(source.dataset_label for source in SUITE_INVENTORY_SOURCES)
SUITE_AGGREGATE_DATASETS: tuple[str, ...] = tuple(source.dataset_label for source in SUITE_AGGREGATE_SOURCES)

_SOURCE_BY_GROUP = {source.source_group: source for source in SUITE_INVENTORY_SOURCES}
_SOURCE_BY_SLUG = {source.slug: source for source in SUITE_INVENTORY_SOURCES}


def normalize_source_group(value: str | None) -> str:
    return str(value or "").strip().lower()


def suite_source_by_group(source_group: str | None) -> SuiteSourceSpec | None:
    return _SOURCE_BY_GROUP.get(normalize_source_group(source_group))


def suite_source_by_slug(slug: str | None) -> SuiteSourceSpec | None:
    return _SOURCE_BY_SLUG.get(str(slug or "").strip())


def is_suite_inventory_source_group(source_group: str | None) -> bool:
    return suite_source_by_group(source_group) is not None


def is_suite_aggregate_source_group(source_group: str | None) -> bool:
    spec = suite_source_by_group(source_group)
    return bool(spec and spec.aggregate_score)


def suite_benchmark_roster() -> list[str]:
    return [source.dataset_label for source in SUITE_AGGREGATE_SOURCES]


def suite_experiment_languages(source: SuiteSourceSpec) -> tuple[str, ...]:
    if source.validation_scope != "multilingual_exec":
        return source.languages
    return tuple(
        language
        for language in source.languages
        if language in OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES
    )
