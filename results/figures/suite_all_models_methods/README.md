# Full-Run Summary Figures

This directory contains the materialized full-run summary figures tracked by the public companion repo.

The expected public exports are split as follows.

Paper-facing figures:

- score decomposition
- generalization breakdown
- quality vs robustness
- compact slice composition in `results/figures/dataset_statistics/compact_slice_composition.*`
- evaluation-dimensions overview in `results/figures/dataset_statistics/evaluation_dimensions_overview.*`

Paper-facing figures are generated from the comparable method-master aggregate. `compact_slice_composition` and `evaluation_dimensions_overview` are paper-facing, but they live under `results/figures/dataset_statistics/`.
That comparable aggregate is source-balanced over the seven active atomic source groups; it is not a raw row-weighted collapse over all rows.

Repo/supplement figures:

- overall leaderboard when the paper already keeps `suite_all_models_methods_method_master_leaderboard.*` as the main exact-value leaderboard table
- source/language coverage
- suite functional summary
- per-source breakdown
- per-model breakdown
- per-language breakdown
- attack robustness breakdown
- detection vs utility
- method stability heatmap
- public-only overall leaderboard

Exact-value leaderboard CSV/JSON, run inventories, and other table-shaped artifacts live in the sibling `results/tables/suite_all_models_methods/` directory. In the public repo, the figure directory keeps PNG/PDF renders plus only the non-duplicated sidecar data needed for the remaining figure families. Any CSV/JSON that remains beside a figure in this directory is figure-local rendering data, not the canonical paper exact-value table.

If the paper body keeps `suite_all_models_methods_method_master_leaderboard.*` as the main leaderboard table, treat `suite_all_models_methods_overall_leaderboard.*` as a repo/supplement visualization rather than a second primary paper result.

Regenerate figures with:

```bash
python scripts/render_paper_figures.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --suite all --paper-track generation_time --require-times-new-roman --output-dir results/figures/suite_all_models_methods
```

To refresh the repository-tracked figure files from the already materialized summary JSON/tables in a clean clone, use:

```bash
python scripts/render_materialized_summary_figures.py --summary-dir results/figures/suite_all_models_methods --table-dir results/tables/suite_all_models_methods --output-dir results/figures/suite_all_models_methods
```
