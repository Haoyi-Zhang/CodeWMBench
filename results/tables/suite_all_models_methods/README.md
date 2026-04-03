# Full-Run Summary Tables

This directory contains the materialized full-run summary tables tracked by the public companion repo.

The expected public exports are split as follows.

Paper-facing exact-value table sources:

- `suite_all_models_methods_method_master_leaderboard.*`
- `suite_all_models_methods_method_model_leaderboard.*`
- benchmark-definition summary in `results/tables/dataset_statistics/benchmark_definition_summary.*`

The aggregate leaderboard tables in this directory are source-balanced over the seven active atomic source groups; they are not raw row-weighted collapses over every generated row.

Repo/supplement tables:

- `method_summary.*` as a descriptive all-successful-row rollup
- `model_summary.*`
- `model_method_summary.*` as a descriptive model-by-method rollup
- `method_source_summary.*`
- `method_attack_summary.*`
- `timing_summary.*`
- `suite_all_models_methods_run_inventory.*`
- `suite_all_models_methods_public_only_method_master_leaderboard.*`
- `suite_all_models_methods_upstream_only_leaderboard.*`

These remain companion-repo exact-value artifacts and should not be duplicated as paper figures unless a specific claim requires them. The raw-to-summary figure pipeline writes the exact-value leaderboard artifacts into this sibling table directory so they are not duplicated under `results/figures/suite_all_models_methods/`.

Regenerate tables with:

```bash
python scripts/export_full_run_tables.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --output-dir results/tables/suite_all_models_methods
```
