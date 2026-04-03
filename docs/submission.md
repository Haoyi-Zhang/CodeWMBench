# Submission And Packaging Notes

This document collects packaging and submission notes for `CodeWMBench`. It is intentionally an auxiliary document for paper or artifact workflows; the public repository identity is defined by the main [`README.md`](../README.md).

## Recommended Order

1. Rebuild compact manifests with `python scripts/build_suite_manifests.py`
2. Run `make suite-precheck`
3. Run `make suite-matrix`
4. Regenerate summary figures and tables from the finished matrix output
5. Stage the release bundle with `bash scripts/package_zenodo.sh`
6. Validate the bundle with `python scripts/validate_release_bundle.py --bundle results/release_bundle`

## Bundle Scope

The release bundle is intended to contain:

- source code and active scripts
- public snapshots and compact evaluation slices
- active suite manifests
- documentation needed to rerun the compact workflow
- provenance manifests for the four pinned baseline implementations used in this release

The bundle intentionally excludes:

- local caches and model weights
- raw per-run result trees
- machine-specific environment state
- local logs and transient diagnostics
- non-activity legacy configs and historical result directories

## Public Data And Multilingual Slice

The repository-level suite inventory remains five-language overall, but the active official-runtime multilingual execution slice is `python`, `cpp`, and `java`. Whenever these multilingual sources appear in paper-facing tables or figures, label them explicitly as `HumanEval-X (py/cpp/java slice)` and `MBXP-5lang (py/cpp/java slice)`.

## Paper vs Companion-Repo Assets

Recommended paper tables:

- main leaderboard table: `suite_all_models_methods_method_master_leaderboard.*` (comparable, source-balanced method leaderboard)
- per-model table: `suite_all_models_methods_method_model_leaderboard.*` (comparable, source-balanced per-model leaderboard)
- benchmark-definition table: source, inventory size, compact size, execution slice, scoring status

Recommended companion-repo exact-value tables (not duplicated under figures):

- `method_summary.*` as a descriptive all-successful-row rollup
- `model_summary.*`
- `model_method_summary.*` as a descriptive model-by-method rollup
- `method_source_summary.*`
- `method_attack_summary.*`
- `timing_summary.*`
- `suite_all_models_methods_run_inventory.*`
- `suite_all_models_methods_public_only_method_master_leaderboard.*`
- `suite_all_models_methods_upstream_only_leaderboard.*`

Recommended paper figures:

- score decomposition: `suite_all_models_methods_score_decomposition.*`
- quality vs robustness: `suite_all_models_methods_quality_vs_robustness.*`
- generalization breakdown: `suite_all_models_methods_generalization_breakdown.*`
- compact slice composition: `results/figures/dataset_statistics/compact_slice_composition.*`
- evaluation-dimensions overview: `results/figures/dataset_statistics/evaluation_dimensions_overview.*`

If the main paper already keeps `suite_all_models_methods_method_master_leaderboard.*` as the exact-value leaderboard table, move the visual `overall leaderboard` figure to the companion repository or supplement instead of repeating the same ranking twice in the body. Keep richer diagnostics such as per-source, per-model, per-language, attack, family, and category breakdowns in the companion repository or supplement unless the paper needs one of them for a specific claim. In this release those assets live under `results/figures/suite_all_models_methods/`, `results/figures/dataset_statistics/`, and `results/tables/suite_all_models_methods/`.

## Practical Checklist

- `make suite-precheck` passes on the final compact workflow
- active outputs are cleaned before a fresh rerun with `python scripts/clean_suite_outputs.py --include-full-matrix --include-release-bundle`
- summary figures are regenerated from the current matrix output with `--require-times-new-roman`
- `bash scripts/package_zenodo.sh` produces a clean release bundle
- `python scripts/validate_release_bundle.py --bundle results/release_bundle` passes
- the bundle excludes raw run outputs, caches, and private checkout state
