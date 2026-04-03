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
- provenance manifests for the four official baselines

The bundle intentionally excludes:

- local caches and model weights
- raw per-run result trees
- machine-specific environment state
- local logs and transient diagnostics
- non-activity legacy configs and historical result directories

## Public Data And Multilingual Slice

The repository-level suite inventory remains five-language overall, but the active official-runtime multilingual execution slice is `python`, `cpp`, and `java`. Whenever `HumanEval-X` or `MBXP-5lang` appear in paper-facing tables or figures, label them as the current common-support executable slice.

## Practical Checklist

- `make suite-precheck` passes on the final compact workflow
- active outputs are cleaned before a fresh rerun with `python scripts/clean_suite_outputs.py --include-full-matrix --include-release-bundle`
- summary figures are regenerated from the current matrix output with `--require-times-new-roman`
- `bash scripts/package_zenodo.sh` produces a clean release bundle
- `python scripts/validate_release_bundle.py --bundle results/release_bundle` passes
- the bundle excludes raw run outputs, caches, and private checkout state
