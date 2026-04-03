# Full-Run Summary Figures

This directory is the repository-tracked export target for full-run summary figures.

In a fresh public clone, this directory may contain only this README until:

- a released raw full-run artifact is restored under `results/matrix/`, or
- the compact full matrix is rerun locally and the figure export scripts are executed.

Regenerate figures with:

```bash
python scripts/render_paper_figures.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --suite all --paper-track generation_time --require-times-new-roman --output-dir results/figures/suite_all_models_methods
```
