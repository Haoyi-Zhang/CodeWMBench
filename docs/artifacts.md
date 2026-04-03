# Artifacts And Large Result Files

The public GitHub repository is intentionally lightweight:

- code
- pinned public benchmark snapshots
- active compact collections
- documentation
- summary figures and tables
- light result inventories and manifests

Large raw full-run outputs are **not** stored in git.

The public GitHub repository alone is therefore sufficient for code inspection, dataset statistics, formulas, and compact benchmark definitions, but **not** for reconstructing the raw `112/112 success` full-run result tree until an external raw artifact is published or restored locally.

Until a Zenodo record is minted or a local rerun is restored, the public repository may only contain placeholder README files in the final full-run summary export directories. That is expected: the repository tracks the export targets, not the raw per-run result tree.

## What Stays Out Of Git

The following are not intended for GitHub storage:

- raw `results/matrix/**` per-run trees
- certification and audit outputs
- local caches and model weights
- runtime upstream checkouts
- machine-specific logs

## Recommended Distribution Strategy

- **GitHub**: code, docs, compact data, summary figures/tables, and lightweight inventories
- **Zenodo**: raw full-run result artifact and optional release bundle

## Raw Result Artifact Layout

The raw artifact should restore the finished full-run tree under:

```text
results/matrix/suite_all_models_methods/
```

and, when available, may also include:

```text
results/certifications/
results/audits/
```

## Suggested Public Metadata

For public release, publish:

- a raw artifact manifest
- a checksum file
- the exact model identifiers used
- the exact upstream provenance manifests used for the four official baselines

Before a DOI is minted, the repository only carries templates for the first two files. Replace the `TBD` fields with the actual Zenodo metadata before citing the public artifact in a paper or GitHub release. Until then, Level 2 reproduction should be understood as a documented regeneration path rather than a self-contained raw-results download from this repository alone.

Repository templates for the first two files live under:

- [`artifacts/raw_results_manifest.template.json`](../artifacts/raw_results_manifest.template.json)
- [`artifacts/SHA256SUMS.template.txt`](../artifacts/SHA256SUMS.template.txt)

## Rebuilding Summary Outputs

After downloading the raw artifact, regenerate the summary outputs locally:

```bash
python scripts/refresh_report_metadata.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json
python scripts/render_paper_figures.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --suite all --paper-track generation_time --require-times-new-roman --output-dir results/figures/suite_all_models_methods
python scripts/export_full_run_tables.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --output-dir results/tables/suite_all_models_methods
```

The repository does not distribute model weights or runtime baseline checkouts. Those are fetched on demand from the exact model IDs and pinned upstream manifests.
