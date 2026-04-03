# Full-Run Summary Tables

This directory is the repository-tracked export target for full-run summary tables.

In a fresh public clone, this directory may contain only this README until:

- a released raw full-run artifact is restored under `results/matrix/`, or
- the compact full matrix is rerun locally and the table export scripts are executed.

Regenerate tables with:

```bash
python scripts/export_full_run_tables.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --output-dir results/tables/suite_all_models_methods
```
