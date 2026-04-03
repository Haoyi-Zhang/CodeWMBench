# Reproducing `CodeWMBench`

Start with **Level 1** unless you explicitly need regenerated full-run summaries or an end-to-end GPU rerun. Level 2 only becomes immediately executable after the archival raw-results artifact is published.

## Level 1: Browse Summary Artifacts

No GPU is required.

Use the repository-tracked dataset statistics, documentation, and the materialized summary figures/tables shipped with this snapshot:

- [`results/figures`](../results/figures)
- [`results/tables`](../results/tables)

This is the default reviewer path. It is the fastest way to inspect the benchmark design, dataset statistics, scoring rules, and the repository-tracked summary outputs that ship with the public companion repo.

If you only want to refresh the repository-tracked full-run figure files from the shipped summary JSON/tables, without restoring the raw matrix tree, run:

```bash
python scripts/render_materialized_summary_figures.py --summary-dir results/figures/suite_all_models_methods --table-dir results/tables/suite_all_models_methods --output-dir results/figures/suite_all_models_methods
```

## Level 2: Regenerate Figures And Tables From Released Artifacts

No GPU is required.

Use this path **after** the archival raw-results artifact described in [`docs/artifacts.md`](artifacts.md) has been published.

1. Download the released raw result artifact described in [`docs/artifacts.md`](artifacts.md)
2. Restore the raw result tree under `results/matrix/`
3. Regenerate figures and tables:

```bash
python scripts/refresh_report_metadata.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json
python scripts/render_paper_figures.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --suite all --paper-track generation_time --require-times-new-roman --output-dir results/figures/suite_all_models_methods
python scripts/export_full_run_tables.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --output-dir results/tables/suite_all_models_methods
python scripts/export_dataset_statistics.py
```

This path reproduces the summary outputs without rerunning models. Until a Zenodo record is minted, the repository only ships the raw-artifact manifest/checksum templates and the regeneration commands.

If Times New Roman is unavailable on the machine, you can use `--allow-font-fallback` for a quick inspection pass. Final camera-ready figures should still be rendered with `--require-times-new-roman`.

## Level 3: Rerun The Compact Matrix

GPU access is required.

### Prerequisites

- Python `3.10+`
- CUDA-capable Linux GPU host
- local toolchains for executable validation
- access to the following model identifiers:
  - `Qwen/Qwen2.5-Coder-14B-Instruct`
  - `Qwen/Qwen2.5-Coder-7B-Instruct`
  - `bigcode/starcoder2-7b`
  - `deepseek-ai/deepseek-coder-6.7b-instruct`

### Upstream Baselines

The four official runtime baselines are fetched by pinned upstream provenance:

```bash
bash scripts/fetch_runtime_upstreams.sh all
```

### Local Workflow

```bash
python scripts/clean_suite_outputs.py --include-full-matrix --include-release-bundle
bash scripts/fetch_runtime_upstreams.sh all
python scripts/build_suite_manifests.py
python scripts/audit_benchmarks.py --profile suite
make suite-precheck
make suite-matrix
make suite-monitor
```

### Linux Remote Workflow

See [`docs/remote_linux_gpu.md`](remote_linux_gpu.md) for the recommended upstream fetch, clean, preflight, precheck, and explicit `--run-full` sequence. The remote launcher cleans active outputs by default; use `--no-clean` only if you intentionally want to preserve an existing result tree.

## Notes

- model weights are not distributed with this repository
- pinned upstream baseline checkouts are not stored in git
- the active multilingual official-runtime comparison slice is `python/cpp/java`
- `HumanEval` remains part of the inventory but is not part of the aggregate compact score
