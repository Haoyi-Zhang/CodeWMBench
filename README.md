# CodeWMBench

`CodeWMBench` is an executable benchmark suite for evaluating code watermarking methods under realistic generation-time conditions. The active public release is the **TOSEM-compact** slice: four pinned upstream baseline implementations executed through CodeWMBench adapters under a shared benchmark-layer generation policy, four local code models, seven atomic benchmark sources, and the full five-dimensional scorecard headed by `CodeWMScore`.

The public release is organized around a browse-first reviewer path. Start by reading the repository-tracked benchmark definition, dataset statistics, formulas, and summary artifacts. Regenerating figures from raw artifacts and rerunning the compact workflow are documented secondary paths.

## What Is In Scope

The active runtime baseline implementations used in this release are:

- `stone_runtime`
- `sweet_runtime`
- `ewd_runtime`
- `kgw_runtime`

The active local model roster is:

- `Qwen/Qwen2.5-Coder-14B-Instruct`
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `bigcode/starcoder2-7b`
- `deepseek-ai/deepseek-coder-6.7b-instruct`

The active suite inventory contains:

- `HumanEval`
- `HumanEval+`
- `MBPP+`
- `HumanEval-X`
- `MBXP-5lang`
- `crafted_original`
- `crafted_translation`
- `crafted_stress`

The aggregate suite score uses the following seven atomic source groups:

- `HumanEval+`
- `MBPP+`
- `HumanEval-X (py/cpp/java slice)`
- `MBXP-5lang (py/cpp/java slice)`
- `Crafted Original`
- `Crafted Translation`
- `Crafted Stress`

`HumanEval` remains part of the documented inventory but is excluded from the aggregate suite score to avoid double-counting against `HumanEval+`.

`HumanEval-X` and `MBXP-5lang` remain five-language inventory datasets, but the active runtime comparison scores only the `python/cpp/java` common-support execution slice. The same compact execution-slice rule applies to the active crafted sources.

## TOSEM-Compact Slice

The published compact slice keeps the benchmark structure intact while reducing wall-clock cost:

- `HumanEval+`: `164`
- `MBPP+`: `240`
- `HumanEval-X (py/cpp/java slice)`: `120`
- `MBXP-5lang (py/cpp/java slice)`: `120`
- `Crafted Original`: `120`
- `Crafted Translation`: `120`
- `Crafted Stress`: `120`

The compact public and crafted slices are deterministic, versioned inputs stored under [`data/compact/collections`](data/compact/collections) and [`data/compact/crafted`](data/compact/crafted). `data/interim/` is reserved for build-time or diagnostic intermediates and is not part of the active public workflow.

The canonical benchmark-definition table for the public release lives under [`results/tables/dataset_statistics/benchmark_definition_summary.csv`](results/tables/dataset_statistics/benchmark_definition_summary.csv) and [`results/tables/dataset_statistics/benchmark_definition_summary.json`](results/tables/dataset_statistics/benchmark_definition_summary.json).

README-facing figures label the multilingual compact sources as `HumanEval-X (py/cpp/java slice)` and `MBXP-5lang (py/cpp/java slice)` in scope-sensitive form; they should not be read as the full five-language inventories.

![TOSEM-compact slice composition](results/figures/dataset_statistics/compact_slice_composition.png)

## Repository Layout

- `codewmbench/`: orchestration, adapters, scoring, reporting, and benchmark logic
- `configs/`: active baseline configs, suite manifests, and utility configs
- `data/public/`: pinned normalized public benchmark snapshots
- `data/compact/collections/`: active TOSEM-compact benchmark collections
- `data/compact/crafted/`: active crafted benchmark snapshots used to build compact collections
- `docs/`: public documentation, formulas, datasets, artifacts, and reproduction guides
- `results/figures/`: dataset statistics figures plus materialized full-run summary figures
- `results/tables/`: dataset statistics tables plus materialized full-run summary tables
- `scripts/`: data preparation, manifest building, audits, figure export, packaging, and helper entrypoints
- `third_party/`: pinned upstream provenance manifests for the four baseline implementations used in this release

## Scoring

Every final report carries a `summary.scorecard` block with:

- `detection_reliability`
- `robustness`
- `utility`
- `stealth`
- `generalization`
- `CodeWMScore`

`CodeWMScore` is the primary ranking metric, but it does not replace the submetrics. It combines a weighted base score with an execution-quality gate so that high false-positive rates or severe quality collapses cannot hide behind a single strong dimension.

The exact formulas are documented in [`docs/metrics.md`](docs/metrics.md).

![Evaluation dimensions overview](results/figures/dataset_statistics/evaluation_dimensions_overview.png)

## Public Results And Artifacts

In a fresh public clone, the repository guarantees:

- code
- compact benchmark inputs
- dataset statistics figures and tables
- documentation and reproduction scripts
- materialized full-run summary figures and tables

The repository does **not** guarantee the raw `112/112 success` full-run tree. Large raw full-run outputs are distributed outside git via the released artifact path described in [`docs/artifacts.md`](docs/artifacts.md).

- dataset statistics figures live under [`results/figures/dataset_statistics`](results/figures/dataset_statistics)
- dataset statistics tables live under [`results/tables/dataset_statistics`](results/tables/dataset_statistics)
- materialized full-run summary figures live under [`results/figures/suite_all_models_methods`](results/figures/suite_all_models_methods)
- materialized full-run summary tables live under [`results/tables/suite_all_models_methods`](results/tables/suite_all_models_methods)
- canonical paper exact-value tables are [`suite_all_models_methods_method_master_leaderboard.*`](results/tables/suite_all_models_methods) and [`suite_all_models_methods_method_model_leaderboard.*`](results/tables/suite_all_models_methods)
- if the paper body keeps the main exact-value leaderboard table, the visual overall leaderboard should stay in the companion repo or supplement rather than repeating the same ranking twice
- those summary outputs can be regenerated from the released raw artifact or a local rerun
- raw full-run artifacts are documented in [`docs/artifacts.md`](docs/artifacts.md)

Use [`docs/reproduce.md`](docs/reproduce.md) for the canonical three-level reviewer path:

1. browse the repository-tracked summary assets
2. regenerate full-run summaries from a published raw artifact
3. rerun the compact workflow on a GPU host

Level 1 is the default reviewer path. If the archival raw-results artifact has not been published yet, Level 2 should be read as a documented future regeneration path rather than an immediately available download.

## Linux GPU Rerun Quick Start

Install lightweight dependencies:

```bash
pip install -r requirements.txt
```

Build and audit the active compact suite inputs:

```bash
bash scripts/fetch_runtime_upstreams.sh all
python scripts/build_suite_manifests.py
python scripts/audit_benchmarks.py --profile suite
python scripts/audit_full_matrix.py --manifest configs/matrices/suite_all_models_methods.json --profile suite_all_models_methods --skip-hf-access
```

For a clean rerun, clear transient outputs first:

```bash
python scripts/clean_suite_outputs.py --include-full-matrix --include-release-bundle
```

Run the two-stage compact precheck:

```bash
make suite-precheck
```

Run the compact full matrix:

```bash
make suite-matrix
```

Watch live progress:

```bash
make suite-monitor
```

Regenerate final full-run summary figures from finished raw results:

```bash
python scripts/render_paper_figures.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --suite all --paper-track generation_time --require-times-new-roman --output-dir results/figures/suite_all_models_methods
```

Export full-run summary tables from finished reports:

```bash
python scripts/export_full_run_tables.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --output-dir results/tables/suite_all_models_methods
```

Export repository-tracked dataset statistics:

```bash
python scripts/export_dataset_statistics.py
```

## Reproduction Levels

- [`docs/reproduce.md`](docs/reproduce.md): step-by-step reproduction paths
- [`docs/datasets.md`](docs/datasets.md): dataset inventory, compact slice rules, and curation notes
- [`docs/metrics.md`](docs/metrics.md): mathematical score definitions
- [`docs/baselines.md`](docs/baselines.md): pinned upstream baseline provenance and fetch rules
- [`docs/artifacts.md`](docs/artifacts.md): raw artifact distribution policy
- [`docs/remote_linux_gpu.md`](docs/remote_linux_gpu.md): Linux GPU rerun workflow

## Provenance And Fetch Policy

- model weights are **not** distributed in this repository
- pinned upstream baseline checkouts are **not** vendored here unless redistributable and explicitly packaged
- model weights are pulled from Hugging Face by exact model identifier
- baseline implementations are fetched from pinned upstream repositories using the manifests in [`third_party`](third_party), while orchestration, local model loading, and decoding policy remain benchmark-controlled
- upstream provenance does not imply uniform redistribution permission; license status is tracked per manifest

The active compact workflow assumes reproducible local-model execution and project-native adapters around the pinned upstream baseline logic. API-backed execution is not part of the current public benchmark path.
